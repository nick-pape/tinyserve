"""Industry-standard MoE expert cache benchmark.

Methodology based on HOBBIT (arxiv 2411.01433), ExpertFlow (arxiv 2410.17954),
Routing Consistency Study (arxiv 2505.16056), and DuoServe-MoE (arxiv 2509.07379).

Measures:
- Hit rate on diverse prompts (not repeated text)
- Cold start vs warm hit rates
- Domain shift impact on cache
- Per-layer hit rate distribution
- Miss latency distribution (p50/p95/p99)
- Expert frequency distribution (Zipf analysis)
- Unique experts per decode step

Usage:
    python -m scripts.cache_benchmark
    python -m scripts.cache_benchmark --phases cold,sustained,shift
    python -m scripts.cache_benchmark --json results.json
"""

import argparse
import json
import statistics
import time

import torch

from scripts.prompts import (
    COLD_START,
    CODE_PROMPTS,
    CONVERSATION_PROMPTS,
    CREATIVE_PROMPTS,
    DOMAIN_MAP,
    DOMAIN_SHIFTS,
    MATH_PROMPTS,
    MULTI_TURN,
    MULTILINGUAL_PROMPTS,
)


def _load_model(model_id, cache_policy="lru", attn="sdpa"):
    from transformers import AutoTokenizer

    from tinyserve.offload import load_and_offload

    model = load_and_offload(model_id, attn_implementation=attn, cache_policy=cache_policy)
    tok = AutoTokenizer.from_pretrained(model_id)
    return model, tok


def _get_cache(model):
    for p in getattr(model, "_offload_pipelines", []):
        if p.cache is not None:
            return p.cache
    return None


def _has_layer_stats(cache) -> bool:
    return getattr(cache, "get_layer_stats", None) is not None


def _has_expert_frequencies(cache) -> bool:
    return getattr(cache, "get_expert_frequencies", None) is not None


def _generate(model, tok, prompt, max_tokens=50):
    """Generate tokens, return (output_text, n_generated, elapsed_s)."""
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_gen = out.shape[1] - inp["input_ids"].shape[1]
    text = tok.decode(out[0], skip_special_tokens=True)
    return text, n_gen, elapsed


def _collect_stats(cache):
    """Collect all stats from cache and return as dict."""
    total_h = cache.hits
    total_m = cache.misses
    total = total_h + total_m
    hr = total_h / total if total > 0 else 0.0

    result = {
        "total_hits": total_h,
        "total_misses": total_m,
        "hit_rate": round(hr, 4),
    }

    if _has_layer_stats(cache):
        layer_stats = cache.get_layer_stats()
        per_layer_hr = {}
        for li, s in layer_stats.items():
            per_layer_hr[li] = s["hit_rate"]
        result["per_layer_hit_rate"] = {str(k): round(v, 4) for k, v in per_layer_hr.items()}

        all_latencies = []
        for li, s in layer_stats.items():
            all_latencies.extend(s.get("miss_latency_ms", []))

        latency_stats = {}
        if all_latencies:
            all_latencies.sort()
            n = len(all_latencies)
            latency_stats = {
                "p50_ms": all_latencies[n // 2],
                "p95_ms": all_latencies[int(n * 0.95)],
                "p99_ms": all_latencies[int(n * 0.99)],
                "mean_ms": statistics.mean(all_latencies),
            }
        result["miss_latency"] = latency_stats
    else:
        result["per_layer_hit_rate"] = {}
        result["miss_latency"] = {}

    if _has_expert_frequencies(cache):
        freq = cache.get_expert_frequencies()
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top_experts = sorted_freq[:20]
        cold_experts = sorted_freq[-20:] if len(sorted_freq) > 20 else []
        result["top_20_experts"] = [(f"L{k[0]}E{k[1]}", v) for k, v in top_experts]
        result["cold_20_experts"] = [(f"L{k[0]}E{k[1]}", v) for k, v in cold_experts]
        result["unique_experts_seen"] = len(freq)
        result["total_possible_experts"] = 24 * 32
    else:
        result["top_20_experts"] = []
        result["cold_20_experts"] = []
        result["unique_experts_seen"] = 0
        result["total_possible_experts"] = 24 * 32

    return result


def _reset_stats(cache):
    """Reset cache stats, using reset_stats() if available, else manual reset."""
    if getattr(cache, "reset_stats", None) is not None:
        cache.reset_stats()
    else:
        cache.hits = 0
        cache.misses = 0


def run_cold_start(model, tok, cache, gen_tokens=30):
    """Phase 1: Sequential diverse prompts — each prompt starts with a truly empty cache."""
    print("\n=== Phase 1: Sequential Diverse Prompts (cold cache per prompt) ===")

    results = []
    for prompt in COLD_START:
        if hasattr(cache, "clear"):
            cache.clear()
        else:
            _reset_stats(cache)
        text, n, elapsed = _generate(model, tok, prompt, max_tokens=gen_tokens)
        stats = _collect_stats(cache)
        domain = prompt[:30] + "..."
        print(f"  {domain:<35} HR={stats['hit_rate']:.1%}  {n/elapsed:.1f} tok/s")
        results.append({"prompt": prompt[:50], **stats})

    return results


def run_sustained(model, tok, cache, domain="code", gen_tokens=40, n_prompts=4):
    """Phase 2: Sustained generation in one domain — measure cache warming."""
    prompts = DOMAIN_MAP.get(domain, CODE_PROMPTS)[:n_prompts]
    print(f"\n=== Phase 2: Sustained {domain} ({n_prompts} prompts, {gen_tokens} tok each) ===")
    _reset_stats(cache)

    results = []
    for i, prompt in enumerate(prompts):
        pre_hits = cache.hits
        pre_misses = cache.misses
        text, n, elapsed = _generate(model, tok, prompt, max_tokens=gen_tokens)
        phase_hits = cache.hits - pre_hits
        phase_misses = cache.misses - pre_misses
        phase_total = phase_hits + phase_misses
        hr = phase_hits / phase_total if phase_total > 0 else 0
        print(f"  Prompt {i+1}/{n_prompts}: HR={hr:.1%}  {n/elapsed:.1f} tok/s  ({phase_hits}h/{phase_misses}m)")
        results.append({"prompt_idx": i, "hit_rate": round(hr, 4), "tps": round(n/elapsed, 1)})

    overall = _collect_stats(cache)
    print(f"  Overall: HR={overall['hit_rate']:.1%}")
    return {"domain": domain, "per_prompt": results, "overall": overall}


def run_domain_shift(model, tok, cache, warmup_domain, shift_domain,
                     warmup_prompts=2, shift_prompts=2, gen_tokens=30):
    """Phase 3: Domain shift — warm on one domain, measure hit rate after switching."""
    print(f"\n=== Phase 3: Domain Shift ({warmup_domain} -> {shift_domain}) ===")

    # Warmup phase
    _reset_stats(cache)
    warmup = DOMAIN_MAP[warmup_domain][:warmup_prompts]
    for p in warmup:
        _generate(model, tok, p, max_tokens=gen_tokens)
    warmup_stats = _collect_stats(cache)
    print(f"  Warmup ({warmup_domain}): HR={warmup_stats['hit_rate']:.1%}")

    # Shift phase — DON'T reset cache, just reset stats
    _reset_stats(cache)
    shift = DOMAIN_MAP[shift_domain][:shift_prompts]
    for p in shift:
        _generate(model, tok, p, max_tokens=gen_tokens)
    shift_stats = _collect_stats(cache)
    print(f"  Shift  ({shift_domain}): HR={shift_stats['hit_rate']:.1%}")
    print(f"  Delta: {shift_stats['hit_rate'] - warmup_stats['hit_rate']:+.1%}")

    return {
        "warmup_domain": warmup_domain,
        "shift_domain": shift_domain,
        "warmup_hr": warmup_stats["hit_rate"],
        "shift_hr": shift_stats["hit_rate"],
        "delta": round(shift_stats["hit_rate"] - warmup_stats["hit_rate"], 4),
    }


def run_per_layer_analysis(model, tok, cache, gen_tokens=50):
    """Phase 4: Per-layer hit rate analysis — which layers are hardest to cache?"""
    print("\n=== Phase 4: Per-Layer Hit Rate Analysis ===")

    if not _has_layer_stats(cache):
        print("  SKIPPED: cache does not support get_layer_stats() yet")
        return {"skipped": True, "reason": "get_layer_stats not available"}

    _reset_stats(cache)

    for prompt in COLD_START[:4]:
        _generate(model, tok, prompt, max_tokens=gen_tokens)

    stats = _collect_stats(cache)
    layer_hrs = stats["per_layer_hit_rate"]

    print(f"  {'Layer':>6}  {'HR%':>6}  bar")
    print(f"  {'---':>6}  {'---':>6}  {'---'*10}")
    for li in sorted(layer_hrs.keys(), key=int):
        hr = layer_hrs[li]
        bar_len = int(hr * 30)
        bar = "#" * bar_len
        print(f"  {li:>6}  {hr:>5.1%}  {bar}")

    worst = sorted(layer_hrs.items(), key=lambda x: x[1])[:5]
    best = sorted(layer_hrs.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Worst 5: {', '.join(f'L{k}={v:.1%}' for k, v in worst)}")
    print(f"  Best  5: {', '.join(f'L{k}={v:.1%}' for k, v in best)}")

    return {"per_layer_hr": layer_hrs, "worst_5": worst, "best_5": best}


def run_expert_frequency(model, tok, cache, gen_tokens=50):
    """Phase 5: Expert frequency distribution — is it Zipf-like?"""
    print("\n=== Phase 5: Expert Frequency Distribution ===")

    if not _has_expert_frequencies(cache):
        print("  SKIPPED: cache does not support get_expert_frequencies() yet")
        return {"skipped": True, "reason": "get_expert_frequencies not available"}

    _reset_stats(cache)

    for prompt in COLD_START + CODE_PROMPTS[:2] + CREATIVE_PROMPTS[:2]:
        _generate(model, tok, prompt, max_tokens=gen_tokens)

    stats = _collect_stats(cache)
    freq = cache.get_expert_frequencies()

    counts = sorted(freq.values(), reverse=True)
    total_accesses = sum(counts)
    top_10_pct = sum(counts[:len(counts)//10]) / total_accesses if total_accesses > 0 else 0
    top_25_pct = sum(counts[:len(counts)//4]) / total_accesses if total_accesses > 0 else 0

    all_experts = set()
    for li in range(24):
        for ei in range(32):
            all_experts.add((li, ei))
    accessed = set(freq.keys())
    never_accessed = len(all_experts - accessed)

    print(f"  Total expert accesses: {total_accesses}")
    print(f"  Unique experts accessed: {len(freq)}/{24*32} ({len(freq)/(24*32):.1%})")
    print(f"  Never accessed: {never_accessed}")
    print(f"  Top 10% of experts handle {top_10_pct:.1%} of accesses")
    print(f"  Top 25% of experts handle {top_25_pct:.1%} of accesses")

    return {
        "total_accesses": total_accesses,
        "unique_experts": len(freq),
        "never_accessed": never_accessed,
        "top_10pct_share": round(top_10_pct, 4),
        "top_25pct_share": round(top_25_pct, 4),
        "frequency_distribution": counts[:50],
    }


def main():
    parser = argparse.ArgumentParser(description="MoE expert cache benchmark")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--phases", default="cold,sustained,shift,layers,frequency",
                       help="Comma-separated phases to run")
    parser.add_argument("--policy", default="lru", help="Cache policy")
    parser.add_argument("--gen-tokens", type=int, default=30)
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    phases = args.phases.split(",")

    print(f"Loading {args.model}...")
    model, tok = _load_model(args.model, cache_policy=args.policy)
    cache = _get_cache(model)
    if cache is None:
        print("ERROR: No expert cache found on model.")
        return

    print(f"Cache: {cache.capacity} slots, policy={args.policy}")
    print(f"Model: 24 layers x 32 experts, top_k=4, 238 slots = {238/(24*32):.0%} coverage")

    results = {}

    _generate(model, tok, "Hello", max_tokens=5)

    if "cold" in phases:
        results["cold_start"] = run_cold_start(model, tok, cache, args.gen_tokens)

    if "sustained" in phases:
        results["sustained"] = {}
        for domain in ["code", "math", "creative", "multilingual", "conversation"]:
            results["sustained"][domain] = run_sustained(model, tok, cache, domain, args.gen_tokens)

    if "shift" in phases:
        results["domain_shifts"] = []
        for warmup, shift in DOMAIN_SHIFTS:
            r = run_domain_shift(model, tok, cache, warmup, shift, gen_tokens=args.gen_tokens)
            results["domain_shifts"].append(r)

    if "layers" in phases:
        results["per_layer"] = run_per_layer_analysis(model, tok, cache, args.gen_tokens)

    if "frequency" in phases:
        results["expert_frequency"] = run_expert_frequency(model, tok, cache, args.gen_tokens)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "cold_start" in results:
        cold_hrs = [r["hit_rate"] for r in results["cold_start"]]
        print(f"  Cold start HR: {min(cold_hrs):.1%} - {max(cold_hrs):.1%} (mean {statistics.mean(cold_hrs):.1%})")
    if "sustained" in results:
        for domain, r in results["sustained"].items():
            print(f"  Sustained {domain}: HR={r['overall']['hit_rate']:.1%}")
    if "domain_shifts" in results:
        for r in results["domain_shifts"]:
            print(f"  Shift {r['warmup_domain']}->{r['shift_domain']}: {r['shift_hr']:.1%} (delta {r['delta']:+.1%})")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
