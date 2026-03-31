"""Test Qwen 3.5-122B GGUF Q4_K — go straight to load_from_gguf."""
import sys, os, time, torch, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG = "benchmarks/qwen_122b_test_20260331.txt"
def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f: f.write(m + "\n")

def main():
    with open(LOG, "w"): pass
    log("=== Qwen 3.5-122B GGUF Q4_K Test ===\n")
    
    import glob
    shards = sorted(glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.5-122B-A10B-GGUF/snapshots/*/Q4_K_S/*.gguf")
    ))
    log(f"Shards: {len(shards)}, total {sum(os.path.getsize(s) for s in shards)/1e9:.1f} GB")
    log(f"RAM: {os.popen('free -g').read().split()[9]}G available")
    log(f"GPU: {torch.cuda.mem_get_info(0)[0]/1e9:.1f}/{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    
    from tinyserve.gguf_loader import load_from_gguf
    
    log("\nAttempting load_from_gguf (disk_offload=True)...")
    t0 = time.time()
    try:
        model = load_from_gguf(shards[0], device="cuda", disk_offload=True, model_id="Qwen/Qwen3.5-122B-A10B")
        log(f"Loaded in {time.time()-t0:.1f}s")
        
        if hasattr(model, '_offload_pipelines'):
            p = model._offload_pipelines[0]
            cache = p.cache
            log(f"Expert cache: {cache.capacity if cache else 'None'} slots")
            log(f"CPU-on-miss: {p.cpu_on_miss}")
        
        # Try generation
        log("\nAttempting generation...")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-122B-A10B")
        inp = tok("Hello", return_tensors="pt").to("cuda")
        
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=5, do_sample=False)
        elapsed = time.perf_counter() - t0
        n = out.shape[1] - inp["input_ids"].shape[1]
        text = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        log(f"{n} tokens in {elapsed:.1f}s = {n/elapsed:.2f} tok/s")
        log(f"Output: {text}")
        
    except Exception as e:
        elapsed = time.time() - t0
        log(f"\nFailed after {elapsed:.1f}s: {type(e).__name__}: {str(e)[:300]}")
        log(traceback.format_exc()[-500:])
    
    log("\nDone.")

if __name__ == "__main__":
    main()
