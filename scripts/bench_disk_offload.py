#!/usr/bin/env python3
"""Benchmark disk_offload mode with all optimizations. Single process, no OOM."""
import gc
import sys
import time
import traceback

sys.path.insert(0, ".")
import torch

LOG = "/tmp/disk_offload_bench.log"
BASE = "The history of artificial intelligence began in antiquity. " * 200


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main():
    with open(LOG, "w") as f:
        f.write("")

    log("=== Disk Offload Benchmark (all optimizations) ===")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB free")

    import psutil
    log(f"RAM: {psutil.virtual_memory().available/1e9:.1f} GB available")

    from transformers import AutoTokenizer
    from tinyserve.offload import load_and_offload

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    log("Loading with disk_offload=True...")
    try:
        model = load_and_offload(
            "openai/gpt-oss-20b",
            cache_capacity=0,
            gpu_memory_utilization=0.90,
            disk_offload=True,
        )
    except Exception:
        log(f"LOAD FAILED:\n{traceback.format_exc()}")
        return

    slots = 0
    ram_slots = 0
    for p in model._offload_pipelines:
        if p.cache:
            slots = p.cache.capacity
        if p.ram_cache:
            ram_slots = p.ram_cache.num_slots
        break
    log(f"GPU slots: {slots}, RAM slots: {ram_slots}")
    log(f"VRAM free: {torch.cuda.mem_get_info(0)[0]/1e6:.0f} MB")
    log(f"RAM available: {psutil.virtual_memory().available/1e9:.1f} GB")

    # Warmup — generate 100 tokens to fill RAM cache with most experts
    log("Warmup (filling RAM cache with 100 tokens)...")
    inp = tok("Hello world " * 10, return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=100, do_sample=False)
    warmup_time = time.perf_counter() - t0
    log(f"Warmup: {warmup_time:.1f}s")

    # Check RAM cache fill
    for p in model._offload_pipelines:
        if p.ram_cache:
            log(f"RAM cache: {len(p.ram_cache._lru)}/{p.ram_cache.num_slots} slots filled, HR={p.ram_cache.hit_rate:.1%}")
            break

    # Benchmark
    log(f"{'phase':>8s} {'ctx':>5s} {'gen':>4s} {'tok/s':>7s} {'HR%':>5s}")
    log("-" * 35)

    # Cold (first run already done above, now measure warm)
    for label, prompt, n_gen in [
        ("warm-10", "The capital of France is", 10),
        ("warm-40", "Explain quantum computing:", 40),
        ("ctx-100", BASE[:400], 20),
        ("ctx-500", BASE[:2000], 20),
        ("ctx-1K", BASE, 20),
    ]:
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        plen = inp["input_ids"].shape[1]
        for p in model._offload_pipelines:
            if p.cache:
                p.cache.reset_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=n_gen, do_sample=False)
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        n = out.shape[1] - plen
        hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
        misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
        hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
        log(f"{label:>8s} {plen:>5d} {n:>4d} {n/el:>6.1f} {hr:>4.0f}%")

    log("Done.")


if __name__ == "__main__":
    main()
