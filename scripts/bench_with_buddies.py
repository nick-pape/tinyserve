"""Benchmark with buddy tables enabled — measure buddy substitution impact."""
import json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG = "benchmarks/buddy_bench_20260331.txt"
def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f: f.write(m + "\n")

def main():
    with open(LOG, "w"): pass
    log("=== Benchmark: LFRU + CPU-on-miss + Buddy Tables ===")

    from transformers import AutoTokenizer
    from tinyserve.offload import load_and_offload
    from scripts.prompts import COLD_START, CODE_PROMPTS, CREATIVE_PROMPTS, MULTILINGUAL_PROMPTS, CONVERSATION_PROMPTS

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    model = load_and_offload(
        "openai/gpt-oss-20b",
        attn_implementation="sdpa",
        buddy_table_path="benchmarks/buddy_tables_gptoss20b.json",
    )

    cache = next(p.cache for p in model._offload_pipelines if p.cache is not None)
    log(f"Cache: {cache.capacity} slots, policy=lfru, buddy tables=loaded")

    prompts_by_domain = {
        "cold": COLD_START[:4],
        "code": CODE_PROMPTS[:4],
        "creative": CREATIVE_PROMPTS[:4],
        "multilingual": MULTILINGUAL_PROMPTS[:4],
        "conversation": CONVERSATION_PROMPTS[:4],
    }

    # Warmup
    inp = tok("Hello", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=3, do_sample=False)

    log(f"\n{'Domain':<15} {'HR%':>6} {'tok/s':>7}")
    log("-" * 32)

    results = {}
    for domain, prompts in prompts_by_domain.items():
        cache.reset_stats()
        total_tok = 0
        t0 = time.perf_counter()
        for p in prompts:
            inp = tok(p, return_tensors="pt", truncation=True, max_length=256).to("cuda")
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=25, do_sample=False)
            total_tok += out.shape[1] - inp["input_ids"].shape[1]
        elapsed = time.perf_counter() - t0
        hr = cache.hits / (cache.hits + cache.misses) if (cache.hits + cache.misses) > 0 else 0
        tps = total_tok / elapsed
        log(f"{domain:<15} {hr:>5.1%} {tps:>6.1f}")
        results[domain] = {"hr": round(hr, 4), "tps": round(tps, 1)}

    with open("benchmarks/buddy_bench_20260331.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nDone.")

if __name__ == "__main__":
    main()
