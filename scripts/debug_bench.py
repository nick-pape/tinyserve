#!/usr/bin/env python3
"""Debug benchmark — captures everything regardless of success/failure.

Writes to /tmp/debug_bench.log at every step. If anything crashes,
the log shows exactly where and why.
"""
import gc
import os
import sys
import time
import traceback

sys.path.insert(0, ".")
LOG = "/tmp/debug_bench.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    with open(LOG, "w") as f:
        f.write("")

    log("=== DEBUG BENCHMARK START ===")

    # Step 1: Environment
    log("Step 1: Environment")
    try:
        import torch
        log(f"  torch: {torch.__version__}")
        log(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"  GPU: {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_memory
            free = torch.cuda.mem_get_info(0)[0]
            log(f"  VRAM: {free/1e9:.1f} / {total/1e9:.1f} GB free")
        else:
            log("  NO CUDA — aborting GPU tests")
            return

        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if r.stdout.strip():
            log(f"  GPU processes: {r.stdout.strip()}")
        else:
            log("  GPU processes: none (clean)")

        import psutil
        ram = psutil.virtual_memory()
        log(f"  RAM: {ram.available/1e9:.1f} / {ram.total/1e9:.1f} GB available")
    except Exception:
        log(f"  ERROR: {traceback.format_exc()}")

    # Step 2: Import tinyserve
    log("Step 2: Import tinyserve")
    try:
        from tinyserve.offload import load_and_offload
        from transformers import AutoTokenizer
        log("  OK")
    except Exception:
        log(f"  ERROR: {traceback.format_exc()}")
        return

    # Step 3: Load model (eager, no KV cache — simplest config)
    log("Step 3: Load model (eager, no KV, gpu_util=0.85)")
    try:
        t0 = time.perf_counter()
        model = load_and_offload(
            "openai/gpt-oss-20b",
            cache_capacity=0,
            gpu_memory_utilization=0.90,
        )
        load_time = time.perf_counter() - t0
        log(f"  Loaded in {load_time:.1f}s")

        slots = 0
        for p in model._offload_pipelines:
            if p.cache:
                slots = p.cache.capacity
                break
        log(f"  Expert slots: {slots}")

        free = torch.cuda.mem_get_info(0)[0]
        log(f"  VRAM free after load: {free/1e9:.2f} GB")
    except Exception:
        log(f"  LOAD ERROR: {traceback.format_exc()}")
        return

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    # Step 4: Short decode (10 tokens)
    log("Step 4: Short decode (10 tokens)")
    try:
        inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
        for p in model._offload_pipelines:
            if p.cache:
                p.cache.reset_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        n = out.shape[1] - inputs["input_ids"].shape[1]
        hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
        misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
        hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
        text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[:60]
        log(f"  {n} tok in {el:.2f}s = {n/el:.1f} tok/s, HR={hr:.0f}%")
        log(f"  Output: {text}")
    except Exception:
        log(f"  ERROR: {traceback.format_exc()}")

    # Step 5: Warm cache decode (40 tokens)
    log("Step 5: Warm cache decode (40 tokens)")
    try:
        # Warmup
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20, do_sample=False)
        for p in model._offload_pipelines:
            if p.cache:
                p.cache.reset_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        n = out.shape[1] - inputs["input_ids"].shape[1]
        hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
        misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
        hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
        log(f"  {n} tok in {el:.2f}s = {n/el:.1f} tok/s, HR={hr:.0f}%")
    except Exception:
        log(f"  ERROR: {traceback.format_exc()}")

    # Step 6: Context scaling (eager, increasing prompt sizes)
    log("Step 6: Context scaling (eager attention)")
    base = "The history of artificial intelligence. " * 200
    for ctx_target in [50, 200, 500, 1000]:
        try:
            inp = tok(base, return_tensors="pt", truncation=True, max_length=ctx_target).to("cuda")
            plen = inp["input_ids"].shape[1]
            for p in model._offload_pipelines:
                if p.cache:
                    p.cache.reset_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=10, do_sample=False)
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            n = out.shape[1] - plen
            hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
            misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
            hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
            free = torch.cuda.mem_get_info(0)[0]
            log(f"  ctx={plen:5d}: {n} tok, {n/el:.1f} tok/s, HR={hr:.0f}%, VRAM_free={free/1e6:.0f}MB")
        except torch.cuda.OutOfMemoryError:
            log(f"  ctx={ctx_target:5d}: OOM")
            torch.cuda.empty_cache()
            break
        except Exception:
            log(f"  ctx={ctx_target:5d}: ERROR: {traceback.format_exc()}")
            break

    # Step 7: Try FlexAttention
    log("Step 7: FlexAttention test")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    try:
        log("  Loading with attn_implementation='flex', max_seq_len=4096, FP8 KV...")
        model_flex = load_and_offload(
            "openai/gpt-oss-20b",
            cache_capacity=0,
            max_seq_len=4096,
            kv_dtype=torch.float8_e4m3fn,
            gpu_memory_utilization=0.90,
            attn_implementation="flex",
        )
        kv = getattr(model_flex, "_kv_cache", None)
        slots = 0
        for p in model_flex._offload_pipelines:
            if p.cache:
                slots = p.cache.capacity
                break
        kv_mb = kv.vram_bytes / 1e6 if kv else 0
        free = torch.cuda.mem_get_info(0)[0]
        log(f"  Loaded. KV={kv_mb:.0f}MB, experts={slots}, VRAM_free={free/1e6:.0f}MB")

        # Warmup (compile)
        log("  Compiling (warmup)...")
        inp = tok("Hello", return_tensors="pt").to("cuda")
        if kv:
            kv.reset()
        with torch.no_grad():
            model_flex.generate(**inp, max_new_tokens=5, do_sample=False, past_key_values=kv)
        log("  Compiled.")

        # Bench
        for ctx_target in [10, 100, 500, 1000, 2000, 3000]:
            if kv and ctx_target > kv.max_seq_len - 20:
                break
            try:
                inp = tok(base, return_tensors="pt", truncation=True, max_length=ctx_target).to("cuda")
                plen = inp["input_ids"].shape[1]
                if kv:
                    kv.reset()
                for p in model_flex._offload_pipelines:
                    if p.cache:
                        p.cache.reset_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    kw = {"past_key_values": kv} if kv else {}
                    out = model_flex.generate(**inp, max_new_tokens=10, do_sample=False, **kw)
                torch.cuda.synchronize()
                el = time.perf_counter() - t0
                n = out.shape[1] - plen
                hits = sum(p.cache.hits for p in model_flex._offload_pipelines if p.cache)
                misses = sum(p.cache.misses for p in model_flex._offload_pipelines if p.cache)
                hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
                free = torch.cuda.mem_get_info(0)[0]
                log(f"  flex ctx={plen:5d}: {n} tok, {n/el:.1f} tok/s, HR={hr:.0f}%, VRAM_free={free/1e6:.0f}MB")
            except torch.cuda.OutOfMemoryError:
                log(f"  flex ctx={ctx_target:5d}: OOM")
                torch.cuda.empty_cache()
                break
            except Exception:
                log(f"  flex ctx={ctx_target:5d}: ERROR: {traceback.format_exc()}")
                break

        del model_flex
    except Exception:
        log(f"  FlexAttention ERROR: {traceback.format_exc()}")

    gc.collect()
    torch.cuda.empty_cache()
    log("=== DEBUG BENCHMARK END ===")
    log(f"Full log at: {LOG}")


if __name__ == "__main__":
    main()
