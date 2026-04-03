# Industry-Standard MoE Benchmark Suite

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark suite that measures expert cache performance under realistic, diverse workloads — matching the methodology used by HOBBIT, ExpertFlow, DuoServe-MoE, and the Routing Consistency Study. All existing hit rate claims (97-100%) are invalidated — they were measured on warm cache with repetitive prompts.

**Architecture:** A `scripts/cache_benchmark.py` CLI tool that loads a model once, then runs a sequence of benchmark phases with different prompt sources. Each phase tracks per-layer hit/miss counters and miss latency. Results are saved as JSON for analysis.

**Tech Stack:** PyTorch, existing tinyserve infrastructure, ShareGPT dataset (via HuggingFace datasets), manual diverse prompt corpus.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `tinyserve/generic_store.py` | Modify | Add per-layer hit/miss counters + miss latency tracking |
| `scripts/cache_benchmark.py` | Create | CLI benchmark tool with diverse workloads |
| `scripts/prompts/` | Create | Diverse prompt corpus (code, math, literature, multilingual) |
| `tests/test_cache_stats.py` | Create | Tests for per-layer stat tracking |

---

### Task 1: Per-layer cache statistics

**Files:**
- Modify: `tinyserve/generic_store.py:558-667` (GenericLRUCache)
- Test: `tests/test_cache_stats.py`

Currently, `GenericLRUCache` has global `hits` and `misses` counters. We need per-layer counters and miss latency tracking to diagnose which layers are problematic.

- [ ] **Step 1: Write failing tests**

Create `tests/test_cache_stats.py`:

```python
"""Tests for per-layer cache statistics."""
import time
import torch
import pytest


def _make_cache(capacity=4, num_layers=3, num_experts=8, device="cpu"):
    from tinyserve.generic_store import GenericLRUCache
    expert_bytes = 64
    cache = GenericLRUCache(capacity, expert_bytes, torch.device(device),
                            num_layers=num_layers, num_experts=num_experts)
    return cache


def test_per_layer_hits_misses():
    cache = _make_cache()
    # Miss on layer 0, expert 0
    assert cache.lookup(0, 0) is None
    cache.allocate(0, 0)
    # Hit on layer 0, expert 0
    assert cache.lookup(0, 0) is not None
    # Miss on layer 1, expert 0
    assert cache.lookup(1, 0) is None

    stats = cache.get_layer_stats()
    assert stats[0]["hits"] == 1
    assert stats[0]["misses"] == 1
    assert stats[1]["hits"] == 0
    assert stats[1]["misses"] == 1


def test_per_layer_miss_latency():
    cache = _make_cache()
    # Simulate a miss — lookup returns None, record latency
    cache.lookup(0, 0)
    cache.record_miss_latency(0, 5.0)  # 5ms

    stats = cache.get_layer_stats()
    assert stats[0]["miss_latency_ms"] == [5.0]


def test_reset_layer_stats():
    cache = _make_cache()
    cache.lookup(0, 0)
    cache.reset_stats()
    stats = cache.get_layer_stats()
    assert stats[0]["hits"] == 0
    assert stats[0]["misses"] == 0


def test_expert_frequency_tracking():
    cache = _make_cache()
    # Access expert 3 on layer 0 three times
    for _ in range(3):
        if cache.lookup(0, 3) is None:
            cache.allocate(0, 3)

    freq = cache.get_expert_frequencies()
    assert freq[(0, 3)] == 3  # 1 miss + 2 hits = 3 accesses


def test_unique_experts_per_step():
    cache = _make_cache()
    cache.begin_step()
    cache.lookup(0, 1)
    cache.lookup(0, 2)
    cache.lookup(0, 1)  # duplicate in same step
    step_stats = cache.end_step()
    assert step_stats["unique_experts_accessed"] == 2
    assert step_stats["total_lookups"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cache_stats.py -x -q`
Expected: FAIL — `get_layer_stats`, `record_miss_latency`, `get_expert_frequencies`, `begin_step`, `end_step` don't exist.

- [ ] **Step 3: Add per-layer stats to GenericLRUCache**

In `tinyserve/generic_store.py`, modify `GenericLRUCache.__init__` to add:

```python
# Per-layer statistics
self._layer_hits: dict[int, int] = {}
self._layer_misses: dict[int, int] = {}
self._layer_miss_latencies: dict[int, list[float]] = {}
self._expert_access_count: dict[tuple[int, int], int] = {}
# Per-step tracking
self._step_experts: set[tuple[int, int]] | None = None
self._step_lookups: int = 0
```

Modify `lookup()` to track per-layer stats:

```python
def lookup(self, layer_idx: int, expert_idx: int) -> int | None:
    slot = self._policy.lookup((layer_idx, expert_idx))
    # Per-layer stats
    key = (layer_idx, expert_idx)
    self._expert_access_count[key] = self._expert_access_count.get(key, 0) + 1
    if self._step_experts is not None:
        self._step_experts.add(key)
        self._step_lookups += 1
    if slot is not None:
        self.hits += 1
        self._layer_hits[layer_idx] = self._layer_hits.get(layer_idx, 0) + 1
    else:
        self.misses += 1
        self._layer_misses[layer_idx] = self._layer_misses.get(layer_idx, 0) + 1
    return slot
```

Add new methods:

```python
def record_miss_latency(self, layer_idx: int, latency_ms: float):
    if layer_idx not in self._layer_miss_latencies:
        self._layer_miss_latencies[layer_idx] = []
    self._layer_miss_latencies[layer_idx].append(latency_ms)

def get_layer_stats(self) -> dict[int, dict]:
    layers = set(self._layer_hits.keys()) | set(self._layer_misses.keys())
    result = {}
    for li in sorted(layers):
        h = self._layer_hits.get(li, 0)
        m = self._layer_misses.get(li, 0)
        result[li] = {
            "hits": h,
            "misses": m,
            "hit_rate": h / (h + m) if (h + m) > 0 else 0.0,
            "miss_latency_ms": self._layer_miss_latencies.get(li, []),
        }
    return result

def get_expert_frequencies(self) -> dict[tuple[int, int], int]:
    return dict(self._expert_access_count)

def begin_step(self):
    self._step_experts = set()
    self._step_lookups = 0

def end_step(self) -> dict:
    result = {
        "unique_experts_accessed": len(self._step_experts) if self._step_experts else 0,
        "total_lookups": self._step_lookups,
    }
    self._step_experts = None
    self._step_lookups = 0
    return result

def reset_stats(self):
    self.hits = 0
    self.misses = 0
    self._layer_hits.clear()
    self._layer_misses.clear()
    self._layer_miss_latencies.clear()
    self._expert_access_count.clear()
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_cache_stats.py -x -v`
Expected: all 5 PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/ --ignore=tests/test_hf_models.py --ignore=tests/test_gpu_int4.py -x -q`
Expected: 330+ pass (325 existing + 5 new)

- [ ] **Step 6: Commit**

```bash
git add tinyserve/generic_store.py tests/test_cache_stats.py
git commit -m "feat: per-layer cache statistics — hits, misses, latency, expert frequency

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Diverse prompt corpus

**Files:**
- Create: `scripts/prompts.py`

Industry papers use diverse, multi-domain prompts. We need prompts that exercise different expert activation patterns.

- [ ] **Step 1: Create prompt corpus**

Create `scripts/prompts.py`:

```python
"""Diverse prompt corpus for cache benchmarking.

Covers domains that activate different expert subsets:
- English technical (code, science)
- Mathematics (formulas, proofs)
- Creative writing (fiction, poetry)
- Multilingual (Russian, Chinese, Arabic)
- Conversational (chat, Q&A)
- Domain shifts (switching between the above)

Based on methodology from HOBBIT (arxiv 2411.01433), ExpertFlow (arxiv 2410.17954),
and the Routing Consistency Study (arxiv 2505.16056).
"""

# Phase 1: Cold start — diverse first prompts (no cache warmth)
COLD_START = [
    "Write a Python function that implements quicksort with type hints.",
    "Explain the proof of Fermat's Last Theorem in accessible language.",
    "The rain drummed against the windowpane as Sarah opened the letter.",
    "Опишите принцип работы квантового компьютера простым языком.",
    "请解释量子纠缠的基本原理。",
    "What are the main causes and effects of ocean acidification?",
    "Derive the Euler-Lagrange equation from the principle of least action.",
    "Write a haiku about a programmer debugging at 3am.",
]

# Phase 2: Domain-specific sustained generation
CODE_PROMPTS = [
    "Implement a red-black tree in Rust with insert, delete, and rebalance.",
    "Write a CUDA kernel for matrix multiplication with shared memory tiling.",
    "Design a lock-free concurrent hash map in C++ using atomic operations.",
    "Implement the Raft consensus algorithm in Go with leader election.",
]

MATH_PROMPTS = [
    "Prove that the sum of 1/n^2 from n=1 to infinity equals pi^2/6.",
    "Solve the heat equation on a finite rod with Dirichlet boundary conditions.",
    "Derive the Black-Scholes formula for European call option pricing.",
    "Prove the Banach fixed-point theorem and give three applications.",
]

CREATIVE_PROMPTS = [
    "Write the opening chapter of a noir detective novel set in 1940s Shanghai.",
    "Compose a Shakespearean sonnet about artificial intelligence.",
    "Write a short story about a lighthouse keeper who discovers time travel.",
    "Create a dialogue between Socrates and a modern AI researcher.",
]

MULTILINGUAL_PROMPTS = [
    "Напишите эссе о влиянии Достоевского на мировую литературу.",
    "用中文详细解释深度学习中的注意力机制。",
    "اشرح نظرية النسبية العامة لأينشتاين بالعربية",
    "Erklären Sie die Grundprinzipien der Quantenmechanik auf Deutsch.",
]

CONVERSATION_PROMPTS = [
    "I just got a puppy and it keeps chewing my shoes. What should I do?",
    "Can you help me plan a week-long trip to Japan on a budget?",
    "My code compiles but gives wrong output. Here's the function: def fib(n)...",
    "What's the difference between a latte, cappuccino, and flat white?",
]

# Phase 3: Domain shift sequences — each tuple is (warmup_domain, shift_domain)
DOMAIN_SHIFTS = [
    ("code", "creative"),       # technical → creative
    ("math", "conversation"),   # formal → casual
    ("code", "multilingual"),   # English → non-English
    ("creative", "math"),       # narrative → symbolic
    ("multilingual", "code"),   # non-English → English technical
]

DOMAIN_MAP = {
    "code": CODE_PROMPTS,
    "math": MATH_PROMPTS,
    "creative": CREATIVE_PROMPTS,
    "multilingual": MULTILINGUAL_PROMPTS,
    "conversation": CONVERSATION_PROMPTS,
}

# Phase 4: ShareGPT-style multi-turn (simulated)
MULTI_TURN = [
    [
        "What is a neural network?",
        "How does backpropagation work?",
        "Can you show me a simple implementation in PyTorch?",
        "Now modify it to use batch normalization.",
    ],
    [
        "Tell me about the French Revolution.",
        "What role did Robespierre play?",
        "How did the Reign of Terror end?",
        "Compare it to the Russian Revolution.",
    ],
]
```

- [ ] **Step 2: Commit**

```bash
git add scripts/prompts.py
git commit -m "feat: diverse prompt corpus for cache benchmarking

Covers code, math, creative, multilingual, conversation, domain shifts,
and multi-turn sequences. Based on HOBBIT/ExpertFlow methodology.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Cache benchmark CLI tool

**Files:**
- Create: `scripts/cache_benchmark.py`

This is the main benchmark tool. It loads the model once and runs multiple phases, collecting per-layer stats.

- [ ] **Step 1: Create the benchmark script**

Create `scripts/cache_benchmark.py`:

```python
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
    layer_stats = cache.get_layer_stats()
    freq = cache.get_expert_frequencies()
    total_h = cache.hits
    total_m = cache.misses
    total = total_h + total_m
    hr = total_h / total if total > 0 else 0.0

    # Per-layer hit rates
    per_layer_hr = {}
    for li, s in layer_stats.items():
        per_layer_hr[li] = s["hit_rate"]

    # Miss latency percentiles
    all_latencies = []
    for li, s in layer_stats.items():
        all_latencies.extend(s["miss_latency_ms"])

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

    # Expert frequency distribution (top-20 and bottom-20)
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    top_experts = sorted_freq[:20]
    cold_experts = sorted_freq[-20:] if len(sorted_freq) > 20 else []

    return {
        "total_hits": total_h,
        "total_misses": total_m,
        "hit_rate": round(hr, 4),
        "per_layer_hit_rate": {str(k): round(v, 4) for k, v in per_layer_hr.items()},
        "miss_latency": latency_stats,
        "top_20_experts": [(f"L{k[0]}E{k[1]}", v) for k, v in top_experts],
        "cold_20_experts": [(f"L{k[0]}E{k[1]}", v) for k, v in cold_experts],
        "unique_experts_seen": len(freq),
        "total_possible_experts": 24 * 32,  # GPT-OSS-20B specific
    }


def run_cold_start(model, tok, cache, gen_tokens=30):
    """Phase 1: Cold start — first prompt in each domain, cache empty."""
    print("\n=== Phase 1: Cold Start (diverse first prompts) ===")
    cache.reset_stats()

    results = []
    for prompt in COLD_START:
        cache.reset_stats()
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
    cache.reset_stats()

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
    print(f"\n=== Phase 3: Domain Shift ({warmup_domain} → {shift_domain}) ===")

    # Warmup phase
    cache.reset_stats()
    warmup = DOMAIN_MAP[warmup_domain][:warmup_prompts]
    for p in warmup:
        _generate(model, tok, p, max_tokens=gen_tokens)
    warmup_stats = _collect_stats(cache)
    print(f"  Warmup ({warmup_domain}): HR={warmup_stats['hit_rate']:.1%}")

    # Shift phase — DON'T reset cache, just reset stats
    cache.reset_stats()
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
    cache.reset_stats()

    # Generate on a mix of prompts
    for prompt in COLD_START[:4]:
        _generate(model, tok, prompt, max_tokens=gen_tokens)

    stats = _collect_stats(cache)
    layer_hrs = stats["per_layer_hit_rate"]

    print(f"  {'Layer':>6}  {'HR%':>6}  {'bar'}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*30}")
    for li in sorted(layer_hrs.keys(), key=int):
        hr = layer_hrs[li]
        bar = "█" * int(hr * 30)
        print(f"  {li:>6}  {hr:>5.1%}  {bar}")

    worst = sorted(layer_hrs.items(), key=lambda x: x[1])[:5]
    best = sorted(layer_hrs.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Worst 5: {', '.join(f'L{k}={v:.1%}' for k, v in worst)}")
    print(f"  Best  5: {', '.join(f'L{k}={v:.1%}' for k, v in best)}")

    return {"per_layer_hr": layer_hrs, "worst_5": worst, "best_5": best}


def run_expert_frequency(model, tok, cache, gen_tokens=50):
    """Phase 5: Expert frequency distribution — is it Zipf-like?"""
    print("\n=== Phase 5: Expert Frequency Distribution ===")
    cache.reset_stats()

    for prompt in COLD_START + CODE_PROMPTS[:2] + CREATIVE_PROMPTS[:2]:
        _generate(model, tok, prompt, max_tokens=gen_tokens)

    stats = _collect_stats(cache)
    freq = cache.get_expert_frequencies()

    # Sort by frequency
    counts = sorted(freq.values(), reverse=True)
    total_accesses = sum(counts)
    top_10_pct = sum(counts[:len(counts)//10]) / total_accesses if total_accesses > 0 else 0
    top_25_pct = sum(counts[:len(counts)//4]) / total_accesses if total_accesses > 0 else 0

    # Count zero-access experts
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
        "frequency_distribution": counts[:50],  # top 50
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
    print(f"Model: 24 layers × 32 experts, top_k=4, 238 slots = {238/(24*32):.0%} coverage")

    results = {}

    # Warmup (1 short generation to prime CUDA)
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

    # Summary
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
            print(f"  Shift {r['warmup_domain']}→{r['shift_domain']}: {r['shift_hr']:.1%} (delta {r['delta']:+.1%})")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test that the script imports cleanly**

Run: `python3 -c "from scripts.cache_benchmark import main; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/cache_benchmark.py
git commit -m "feat: industry-standard MoE cache benchmark suite

5 phases: cold start, sustained domain, domain shift, per-layer
analysis, expert frequency distribution. Based on HOBBIT/ExpertFlow
methodology. Outputs JSON for analysis.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Run benchmark and save results

**Files:**
- Save: `benchmarks/cache_benchmark_20260326.json`
- Save: `benchmarks/cache_benchmark_20260326.txt`

- [ ] **Step 1: Create benchmark runner script**

```bash
# Must run as background process to avoid session crash
nohup python3 -m scripts.cache_benchmark \
  --model openai/gpt-oss-20b \
  --policy lru \
  --gen-tokens 30 \
  --json benchmarks/cache_benchmark_20260326.json \
  > benchmarks/cache_benchmark_20260326.txt 2>&1 &
```

- [ ] **Step 2: Wait for completion and review results**

```bash
cat benchmarks/cache_benchmark_20260326.txt
```

- [ ] **Step 3: Commit results**

```bash
git add benchmarks/cache_benchmark_20260326.*
git commit -m "bench: realistic cache benchmark — diverse prompts, per-layer stats

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Update README with honest numbers

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace inflated hit rate claims with real data**

Update the Performance section to include cache benchmark results. Add a note explaining the difference between warm-cache (post-prefill) and cold/diverse workload hit rates.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update benchmark numbers with realistic diverse-workload data

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
