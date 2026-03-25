#!/usr/bin/env python3
"""Long context benchmark with FlexAttention. Writes results incrementally."""
import gc
import subprocess
import sys
import time

sys.path.insert(0, ".")

import torch

MODEL = "openai/gpt-oss-20b"
OUT = "/tmp/long_context_results.txt"
GEN = 20
BASE = "The history of artificial intelligence began in antiquity. " * 200


def w(msg):
    print(msg, flush=True)
    with open(OUT, "a") as f:
        f.write(msg + "\n")


def main():
    with open(OUT, "w") as f:
        f.write("")

    # Wait for GPU
    while True:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if not r.stdout.strip():
            break
        w(f"GPU busy, waiting...")
        time.sleep(30)

    from transformers import AutoTokenizer
    from tinyserve.offload import load_and_offload

    tok = AutoTokenizer.from_pretrained(MODEL)
    w(f"=== Long Context: {MODEL} (FlexAttention + FP8 KV) ===")
    w(f"GPU: {torch.cuda.get_device_name(0)}")

    for max_seq in [32768]:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            model = load_and_offload(
                MODEL, cache_capacity=0, max_seq_len=max_seq,
                kv_dtype=torch.float8_e4m3fn, gpu_memory_utilization=0.88,
                attn_implementation="flex",
            )
        except Exception as e:
            w(f"LOAD FAIL max_seq={max_seq}: {e}")
            continue

        kv = getattr(model, "_kv_cache", None)
        slots = 0
        for p in model._offload_pipelines:
            if p.cache:
                slots = p.cache.capacity
                break
        kv_mb = kv.vram_bytes / 1e6 if kv else 0
        w(f"max_seq={max_seq} KV={kv_mb:.0f}MB experts={slots}")
        w(f"{'ctx':>7s} {'gen':>4s} {'tok/s':>7s} {'ms/tok':>8s} {'HR%':>5s}")
        w("-" * 38)

        # Warmup (includes torch.compile)
        w("Warmup (compiling FlexAttention)...")
        inputs = tok("Hello world", return_tensors="pt").to("cuda")
        if kv:
            kv.reset()
        with torch.no_grad():
            kw = {"past_key_values": kv} if kv else {}
            model.generate(**inputs, max_new_tokens=10, do_sample=False, **kw)
        w("Compiled.")

        for ctx in [10, 50, 100, 500, 1000, 2000, 4000, 8000, 16000, 24000]:
            if ctx > max_seq - GEN - 10:
                break
            inputs = tok(BASE, return_tensors="pt", truncation=True, max_length=ctx).to("cuda")
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
                    out = model.generate(**inputs, max_new_tokens=GEN, do_sample=False, **kw)
                torch.cuda.synchronize()
                el = time.perf_counter() - t0
                n = out.shape[1] - plen
                hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
                misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
                hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
                w(f"{plen:>7d} {n:>4d} {n/el:>6.1f} {el/n*1000:>7.1f} {hr:>4.0f}%")
            except Exception as e:
                w(f"{plen:>7d}  ERR: {str(e)[:50]}")
                break

        del model
        gc.collect()
        torch.cuda.empty_cache()

    w("Done.")


if __name__ == "__main__":
    main()
