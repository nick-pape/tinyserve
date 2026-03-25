#!/usr/bin/env python3
"""Benchmark different KV cache configurations."""
import gc
import sys
import time

import torch

sys.path.insert(0, ".")

from transformers import AutoTokenizer

from tinyserve.offload import load_and_offload

configs = [
    ("No KV (baseline)", 0, torch.bfloat16),
    ("4K BF16", 4096, torch.bfloat16),
    ("4K FP8", 4096, torch.float8_e4m3fn),
    ("32K FP8", 32768, torch.float8_e4m3fn),
]

tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
print(f"{'Config':>15s} {'KV MB':>7s} {'Experts':>8s} {'tok/s':>7s} {'HR%':>5s}")
print("-" * 48)

for label, seq_len, kv_dt in configs:
    gc.collect()
    torch.cuda.empty_cache()
    model = load_and_offload(
        "openai/gpt-oss-20b", cache_capacity=0,
        max_seq_len=seq_len, kv_dtype=kv_dt,
    )
    kv = getattr(model, "_kv_cache", None)
    kv_mb = kv.vram_bytes / 1e6 if kv else 0
    slots = model._offload_pipelines[0].cache.capacity if model._offload_pipelines[0].cache else 0

    inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
    if kv:
        kv.reset()
    with torch.no_grad():
        kw = {"past_key_values": kv} if kv else {}
        model.generate(**inputs, max_new_tokens=10, do_sample=False, **kw)
    if kv:
        kv.reset()
    for p in model._offload_pipelines:
        if p.cache:
            p.cache.reset_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        kw = {"past_key_values": kv} if kv else {}
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False, **kw)
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    n = out.shape[1] - inputs["input_ids"].shape[1]
    hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
    misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
    hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
    print(f"{label:>15s} {kv_mb:>6.0f}M {slots:>8d} {n / el:>6.1f} {hr:>4.0f}%")
    del model
