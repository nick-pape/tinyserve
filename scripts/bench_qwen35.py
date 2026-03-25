#!/usr/bin/env python3
"""Benchmark Qwen3.5-35B-A3B on tinyserve.

Results written to /tmp/qwen35_bench_results.txt.
Run as: nohup python scripts/bench_qwen35.py > /tmp/qwen35_bench.log 2>&1 &
"""
import gc
import sys
import time

sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer

from tinyserve.offload import load_and_offload

MODEL = "Qwen/Qwen3.5-35B-A3B"
OUT = "/tmp/qwen35_bench_results.txt"


def bench_prompt(model, tok, prompt, n_gen, kv=None, label=""):
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    plen = inputs["input_ids"].shape[1]

    if kv:
        kv.reset()
    for p in model._offload_pipelines:
        if p.cache:
            p.cache.reset_stats()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        kw = {"past_key_values": kv} if kv else {}
        out = model.generate(**inputs, max_new_tokens=n_gen, do_sample=False, **kw)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n = out.shape[1] - plen
    hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
    misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
    hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
    text = tok.decode(out[0, plen : plen + 30], skip_special_tokens=True)[:60]
    return {
        "label": label,
        "prompt_tokens": plen,
        "gen_tokens": n,
        "tok_s": n / elapsed,
        "elapsed": elapsed,
        "hit_rate": hr,
        "text": text,
    }


def main():
    results = []
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"=== Qwen3.5-35B-A3B Benchmark on tinyserve ===")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log("")

    # Load model
    log("Loading model...")
    t0 = time.perf_counter()
    try:
        model = load_and_offload(MODEL, cache_capacity=0, cache_policy="lfru")
    except Exception as e:
        log(f"LOAD FAILED: {e}")
        with open(OUT, "w") as f:
            f.write("\n".join(lines))
        return

    load_time = time.perf_counter() - t0
    log(f"Loaded in {load_time:.1f}s")

    tok = AutoTokenizer.from_pretrained(MODEL)
    kv = getattr(model, "_kv_cache", None)
    slots = 0
    for p in model._offload_pipelines:
        if p.cache:
            slots = p.cache.capacity
            break

    log(f"Expert cache: {slots} slots")
    log(f"KV cache: {'yes' if kv else 'no'}")
    log("")

    # Warmup
    log("Warmup...")
    bench_prompt(model, tok, "Hello", 5, kv)
    log("Warmup done.")
    log("")

    # Benchmarks
    prompts = [
        ("Short EN", "The capital of France is", 20),
        ("Medium EN", "Explain quantum computing in simple terms:", 40),
        ("Code", 'def fibonacci(n):\n    """Return nth fib."""', 40),
        ("Long EN", "Write a detailed essay about the history of artificial intelligence:", 60),
        ("Russian", "Столица России — город", 30),
    ]

    header = f"{'Scenario':>15s} {'Prompt':>7s} {'Gen':>5s} {'tok/s':>7s} {'HR%':>5s}"
    log(header)
    log("-" * 44)

    for label, prompt, n_gen in prompts:
        r = bench_prompt(model, tok, prompt, n_gen, kv, label)
        row = f"{r['label']:>15s} {r['prompt_tokens']:>7d} {r['gen_tokens']:>5d} {r['tok_s']:>6.1f} {r['hit_rate']:>4.0f}%"
        log(row)
        results.append(r)

    log("")
    avg = sum(r["tok_s"] for r in results) / len(results)
    log(f"Average: {avg:.1f} tok/s")
    log("")

    # Sample output
    log("Sample output (Medium EN):")
    r = bench_prompt(model, tok, "Explain quantum computing:", 50, kv)
    log(f"  {r['text']}...")
    log("")
    log("Done.")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"Results saved to {OUT}")


if __name__ == "__main__":
    main()
