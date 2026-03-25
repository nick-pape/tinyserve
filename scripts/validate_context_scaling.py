#!/usr/bin/env python3
"""Validate context scaling with GPT-OSS-20B.

Loads model ONCE, then tests progressively longer prompts.
Results written to /tmp/context_scaling_results.txt.

Usage:
    python scripts/validate_context_scaling.py
"""
import gc
import sys
import time

sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer

from tinyserve.offload import load_and_offload

OUT = "/tmp/context_scaling_results.txt"
MODEL = "openai/gpt-oss-20b"
BASE = "The field of artificial intelligence has seen remarkable advances in recent years. "


def run_config(label, max_seq, kv_dt):
    gc.collect()
    torch.cuda.empty_cache()
    try:
        model = load_and_offload(MODEL, cache_capacity=0, max_seq_len=max_seq, kv_dtype=kv_dt)
    except Exception as e:
        return [(label, 0, 0, 0, 0, 0, 0, f"LOAD: {e!s:.30s}")]

    kv = getattr(model, "_kv_cache", None)
    kv_mb = kv.vram_bytes / 1e6 if kv else 0
    slots = 0
    for p in model._offload_pipelines:
        if p.cache:
            slots = p.cache.capacity
            break

    tok = AutoTokenizer.from_pretrained(MODEL)
    results = []

    for mult in [1, 5, 20, 100, 500, 2000]:
        prompt = BASE * mult
        max_len = max_seq if max_seq > 0 else 4096
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=min(max_len - 30, len(prompt))).to("cuda")
        plen = inputs["input_ids"].shape[1]

        if kv:
            kv.reset()
        for p in model._offload_pipelines:
            if p.cache:
                p.cache.reset_stats()

        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                kw = {"past_key_values": kv} if kv else {}
                out = model.generate(**inputs, max_new_tokens=20, do_sample=False, **kw)
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            n = out.shape[1] - plen
            hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
            misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
            hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
            results.append((label, plen, n, n / el, hr, slots, kv_mb, "OK"))
        except Exception as e:
            results.append((label, plen, 0, 0, 0, slots, kv_mb, f"{e!s:.30s}"))
            break

    del model
    return results


def main():
    configs = [
        ("No KV", 0, torch.bfloat16),
        ("4K BF16", 4096, torch.bfloat16),
        ("8K BF16", 8192, torch.bfloat16),
        ("16K BF16", 16384, torch.bfloat16),
        ("8K FP8", 8192, torch.float8_e4m3fn),
        ("32K FP8", 32768, torch.float8_e4m3fn),
    ]

    header = f"{'Config':>10s} {'Prompt':>7s} {'Gen':>4s} {'tok/s':>7s} {'HR%':>5s} {'Slots':>6s} {'KV MB':>7s} {'Status':>8s}"
    sep = "-" * 62

    lines = [header, sep]
    print(header)
    print(sep)

    for label, max_seq, kv_dt in configs:
        rows = run_config(label, max_seq, kv_dt)
        for r in rows:
            lab, plen, n, tps, hr, slots, kvmb, status = r
            line = f"{lab:>10s} {plen:>7d} {n:>4d} {tps:>6.1f} {hr:>4.0f}% {slots:>6d} {kvmb:>6.0f}M {status:>8s}"
            lines.append(line)
            print(line)

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
