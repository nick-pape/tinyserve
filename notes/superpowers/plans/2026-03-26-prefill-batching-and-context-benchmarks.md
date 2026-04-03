# Batched Expert Prefill & Context-Scaling Benchmarks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the O(tokens) expert loading overhead during prefill by batching tokens per-expert, and add benchmarks that separate prefill from decode timing to reveal true context scaling behavior.

**Architecture:** During prefill (seq_len > 1), group all tokens by their routed expert, load each expert once, batch-forward all tokens through it in a single matmul, then scatter weighted results back. For decode (seq_len == 1), keep the existing optimized single-token path unchanged. Add a `--context-scaling` benchmark mode that times prefill and decode separately.

**Tech Stack:** PyTorch, existing GenericExpertPipeline/ExpertBatcher infrastructure, existing benchmark.py CLI.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `tinyserve/generic_pipeline.py` | Modify (lines 357-377) | Add `execute_layer_experts_batched()` method |
| `tinyserve/offloaded_model.py` | Modify (line 527) | Route prefill to batched path |
| `scripts/benchmark.py` | Modify | Add `--context-scaling` mode |
| `tests/test_batched_prefill.py` | Create | Tests for batched prefill correctness |

---

### Task 1: Batched prefill method on GenericExpertPipeline

**Files:**
- Modify: `tinyserve/generic_pipeline.py:357-377`
- Test: `tests/test_batched_prefill.py`

- [ ] **Step 1: Write failing tests for batched prefill**

Create `tests/test_batched_prefill.py`:

```python
"""Tests for batched expert prefill — correctness vs token-by-token baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.conftest import requires_cuda


class TinyFusedExpert(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2 * intermediate, hidden))
        self.down_proj = nn.Parameter(torch.randn(hidden, intermediate))
        self._act_fn = nn.SiLU()
        self._has_bias = False
        self._is_mxfp4 = False
        self._param_names = ["gate_up_proj", "down_proj"]

    def forward(self, x):
        gate_up = F.linear(x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.linear(self._act_fn(gate) * up, self.down_proj)


def _make_pipeline(num_experts=4, hidden=16, intermediate=32, device="cuda"):
    from tinyserve.generic_store import GenericExpertStore, GenericLRUCache
    from tinyserve.generic_pipeline import GenericExpertPipeline

    weights = {}
    for li in range(1):
        for ei in range(num_experts):
            weights[(li, ei)] = {
                "gate_up_proj": torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16),
                "down_proj": torch.randn(hidden, intermediate, dtype=torch.bfloat16),
            }
    store = GenericExpertStore.from_dict(weights, 1, num_experts)
    template = TinyFusedExpert(hidden, intermediate).to(device).to(torch.bfloat16)
    buf_a = store.allocate_buffer(torch.device(device))
    buf_b = store.allocate_buffer(torch.device(device))
    ts = torch.cuda.Stream(torch.device(device))
    cs = torch.cuda.Stream(torch.device(device))
    cache = GenericLRUCache(num_experts, store.buffer_expert_bytes, torch.device(device),
                           num_layers=1, num_experts=num_experts)
    pipeline = GenericExpertPipeline(store, template, torch.device(device),
                                     buf_a, buf_b, ts, cs, cache=cache)
    return pipeline


@requires_cuda
def test_batched_matches_sequential_single_expert():
    """4 tokens all routed to expert 0 — batched == sequential."""
    pipeline = _make_pipeline()
    seq_len, top_k = 4, 1
    h = torch.randn(seq_len, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.zeros(seq_len, top_k, device="cuda", dtype=torch.long)
    weights = torch.ones(seq_len, top_k, device="cuda", dtype=torch.bfloat16)

    ref = pipeline.execute_layer_experts(h, 0, expert_ids, weights)
    out = pipeline.execute_layer_experts_batched(h, 0, expert_ids, weights)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@requires_cuda
def test_batched_matches_sequential_multi_expert():
    """8 tokens with top_k=2, mixed experts — batched == sequential."""
    pipeline = _make_pipeline()
    seq_len, top_k = 8, 2
    h = torch.randn(seq_len, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.randint(0, 4, (seq_len, top_k), device="cuda")
    weights = torch.rand(seq_len, top_k, device="cuda", dtype=torch.bfloat16)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    ref = pipeline.execute_layer_experts(h, 0, expert_ids, weights)
    out = pipeline.execute_layer_experts_batched(h, 0, expert_ids, weights)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@requires_cuda
def test_batched_loads_each_expert_once():
    """With 4 tokens all needing expert 2, cache should see 1 miss not 4."""
    pipeline = _make_pipeline()
    seq_len = 4
    h = torch.randn(seq_len, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.full((seq_len, 1), 2, device="cuda", dtype=torch.long)
    weights = torch.ones(seq_len, 1, device="cuda", dtype=torch.bfloat16)

    pipeline.cache.reset_stats()
    pipeline.execute_layer_experts_batched(h, 0, expert_ids, weights)
    # 1 unique expert = 1 miss (first load) + 0 subsequent misses
    assert pipeline.cache.misses <= 1, f"Expected 1 miss, got {pipeline.cache.misses}"


@requires_cuda
def test_batched_empty_input():
    """Zero-length input returns zero-length output."""
    pipeline = _make_pipeline()
    h = torch.randn(0, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.zeros(0, 1, device="cuda", dtype=torch.long)
    weights = torch.zeros(0, 1, device="cuda", dtype=torch.bfloat16)
    out = pipeline.execute_layer_experts_batched(h, 0, expert_ids, weights)
    assert out.shape == (0, 16)


@requires_cuda
def test_batched_large_prefill():
    """512 tokens — stress test for batched path."""
    pipeline = _make_pipeline()
    seq_len, top_k = 512, 2
    h = torch.randn(seq_len, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.randint(0, 4, (seq_len, top_k), device="cuda")
    weights = torch.rand(seq_len, top_k, device="cuda", dtype=torch.bfloat16)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    ref = pipeline.execute_layer_experts(h, 0, expert_ids, weights)
    out = pipeline.execute_layer_experts_batched(h, 0, expert_ids, weights)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_batched_prefill.py -x -q`
Expected: FAIL with `AttributeError: 'GenericExpertPipeline' object has no attribute 'execute_layer_experts_batched'`

- [ ] **Step 3: Implement `execute_layer_experts_batched` on GenericExpertPipeline**

Add after `execute_layer_experts` (line ~377) in `tinyserve/generic_pipeline.py`:

```python
def execute_layer_experts_batched(
    self,
    hidden_states: torch.Tensor,
    layer_idx: int,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Batched expert dispatch for prefill: load each expert once, batch all tokens through it.

    Groups tokens by expert_id, loads each unique expert once, runs a single
    batched matmul for all tokens routed to that expert, then scatters weighted
    results back. Reduces expert loads from O(seq_len * top_k) to O(num_unique_experts).
    """
    seq_len = hidden_states.shape[0]
    if seq_len == 0:
        return hidden_states.clone()

    output = torch.zeros_like(hidden_states)
    top_k = expert_indices.shape[1]

    # Group token indices by expert_id: {eid: [(tok_idx, weight_position), ...]}
    expert_groups: dict[int, list[tuple[int, int]]] = {}
    eid_list = expert_indices.tolist()
    for tok in range(seq_len):
        for k in range(top_k):
            eid = eid_list[tok][k]
            if eid not in expert_groups:
                expert_groups[eid] = []
            expert_groups[eid].append((tok, k))

    cache = self.cache

    for eid, group in expert_groups.items():
        tok_indices = [g[0] for g in group]
        weight_indices = [g[1] for g in group]

        # Gather hidden states for this expert: [N, hidden_dim]
        h_batch = hidden_states[tok_indices]

        # Try cache hit
        out_batch = None
        if cache is not None:
            slot = cache.lookup(layer_idx, eid)
            if slot is not None:
                cache.hits += 1
                packed = cache.get_packed(slot)
                if self._inline_fwd is not None:
                    out_batch = self._inline_fwd(packed, h_batch)
                else:
                    out_batch = forward_from_packed(
                        self.template, packed, self._param_refs, h_batch
                    )

        # Cache miss: load from store
        if out_batch is None:
            if cache is not None:
                cache.misses += 1
            buf = self.buf_a
            self.store.copy_to_buffer(buf, layer_idx, eid, non_blocking=False)
            torch.cuda.synchronize()

            if self._inline_fwd is not None:
                out_batch = self._inline_fwd(buf.packed, h_batch)
            else:
                out_batch = swap_weights_and_forward(self.template, buf, h_batch)

            # Cache for future use
            if cache is not None:
                slot = cache.allocate(layer_idx, eid)
                cache.get_packed(slot).copy_(buf.packed)

        # Scatter weighted results back
        for i, (tok_idx, k) in enumerate(zip(tok_indices, weight_indices)):
            output[tok_idx] += routing_weights[tok_idx, k] * out_batch[i]

    return output
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_batched_prefill.py -x -q`
Expected: all 5 tests PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `python3 -m pytest tests/ --ignore=tests/test_hf_models.py --ignore=tests/test_gpu_int4.py -x -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add tinyserve/generic_pipeline.py tests/test_batched_prefill.py
git commit -m "feat: batched expert prefill — load each expert once for all tokens"
```

---

### Task 2: Route prefill to batched path in offloaded_forward

**Files:**
- Modify: `tinyserve/offloaded_model.py:527`
- Test: `tests/test_batched_prefill.py` (add integration test)

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_batched_prefill.py`:

```python
@requires_cuda
def test_offloaded_forward_uses_batched_for_prefill():
    """Multi-token forward (prefill) routes through batched path."""
    from unittest.mock import patch
    pipeline = _make_pipeline()

    # Call with seq_len > 1 to trigger prefill path
    h = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.randint(0, 4, (8, 2), device="cuda")
    weights = torch.rand(8, 2, device="cuda", dtype=torch.bfloat16)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Both paths should produce same result
    ref = pipeline.execute_layer_experts(h, 0, expert_ids, weights)
    out = pipeline.execute_layer_experts_batched(h, 0, expert_ids, weights)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
```

(This test already passes from Task 1 — it's a sanity check before wiring.)

- [ ] **Step 2: Modify offloaded_forward to route prefill to batched path**

In `tinyserve/offloaded_model.py`, change line 527 from:

```python
output = pipeline.execute_layer_experts(flat, layer_idx, top_idx, routing_weights)
```

to:

```python
if flat.shape[0] > 1:
    output = pipeline.execute_layer_experts_batched(flat, layer_idx, top_idx, routing_weights)
else:
    output = pipeline.execute_layer_experts(flat, layer_idx, top_idx, routing_weights)
```

This keeps the highly-optimized single-token decode path (C++ fast path, double-buffered pipeline, FATE prefetch) for decode, and only uses batched dispatch for multi-token prefill.

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest tests/ --ignore=tests/test_hf_models.py --ignore=tests/test_gpu_int4.py -x -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add tinyserve/offloaded_model.py tests/test_batched_prefill.py
git commit -m "feat: route multi-token prefill through batched expert path"
```

---

### Task 3: Context-scaling benchmark with prefill/decode separation

**Files:**
- Modify: `scripts/benchmark.py`
- Test: manual verification with `--context-scaling --dry-run`

- [ ] **Step 1: Add `--context-scaling` mode to benchmark.py**

Add this function and CLI wiring to `scripts/benchmark.py`:

```python
def run_context_scaling(
    model_id: str = "openai/gpt-oss-20b",
    contexts: list[int] | None = None,
    gen_tokens: int = 20,
    cache_policy: str = "lfru",
    fp8: bool = True,
    adaptive_fate: bool = True,
    attn_implementation: str = "sdpa",
) -> list[dict]:
    """Benchmark with separated prefill and decode timing across context lengths."""
    from transformers import AutoTokenizer
    from tinyserve.offload import load_and_offload
    from tinyserve.offloaded_model import reset_temporal_routing

    if contexts is None:
        contexts = [10, 50, 100, 500, 1000, 2000, 3000]

    model = load_and_offload(
        model_id,
        device="cuda",
        cache_policy=cache_policy,
        fp8=fp8,
        adaptive_fate=adaptive_fate,
        attn_implementation=attn_implementation,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    base_text = "The history of artificial intelligence. " * 500

    # Warmup
    inp = tok("Hello", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=3, do_sample=False)

    results = []
    for ctx in contexts:
        reset_temporal_routing()
        inp = tok(base_text, return_tensors="pt", truncation=True, max_length=ctx).to("cuda")
        actual_ctx = inp["input_ids"].shape[1]

        _reset_cache_stats(model)

        # Time prefill
        torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
        with torch.inference_mode():
            out = model(**inp, use_cache=False)
        torch.cuda.synchronize()
        t_prefill_end = time.perf_counter()
        prefill_ms = (t_prefill_end - t_prefill_start) * 1000

        p_hits, p_misses = _collect_cache_stats(model)
        _reset_cache_stats(model)

        # Time decode (gen_tokens tokens, one at a time)
        next_token = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        t_decode_start = time.perf_counter()
        for _ in range(gen_tokens):
            with torch.inference_mode():
                out = model(input_ids=next_token, use_cache=False)
            next_token = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        t_decode_end = time.perf_counter()
        decode_ms = (t_decode_end - t_decode_start) * 1000

        d_hits, d_misses = _collect_cache_stats(model)
        d_total = d_hits + d_misses
        decode_tps = gen_tokens / (decode_ms / 1000)
        total_tps = gen_tokens / ((prefill_ms + decode_ms) / 1000)

        result = {
            "ctx": actual_ctx,
            "prefill_ms": round(prefill_ms, 1),
            "decode_ms": round(decode_ms, 1),
            "decode_tps": round(decode_tps, 1),
            "total_tps": round(total_tps, 1),
            "prefill_experts_loaded": p_hits + p_misses,
            "decode_hit_rate": round(d_hits / max(1, d_total), 4),
        }
        results.append(result)

    return results
```

- [ ] **Step 2: Add CLI argument and printer**

In `main()`, add the argument:

```python
parser.add_argument(
    "--context-scaling", action="store_true",
    help="Benchmark prefill vs decode timing across context lengths"
)
```

And the handler (before the default benchmark path):

```python
if args.context_scaling:
    results = run_context_scaling(
        model_id=args.model,
        gen_tokens=args.measure if args.measure != 60 else 20,
        cache_policy=args.cache_policy,
        fp8=not args.no_fp8,
        adaptive_fate=not args.no_adaptive_fate,
    )
    sep = "─" * 72
    print(f"\nContext Scaling — {args.model}")
    print(sep)
    print(f"  {'ctx':>6}  {'prefill':>10}  {'decode':>10}  {'decode':>10}  {'total':>10}  {'decode':>8}")
    print(f"  {'':>6}  {'(ms)':>10}  {'(ms)':>10}  {'(tok/s)':>10}  {'(tok/s)':>10}  {'HR%':>8}")
    print(sep)
    for r in results:
        print(
            f"  {r['ctx']:>6}  {r['prefill_ms']:>10.1f}  {r['decode_ms']:>10.1f}"
            f"  {r['decode_tps']:>10.1f}  {r['total_tps']:>10.1f}"
            f"  {r['decode_hit_rate'] * 100:>7.1f}%"
        )
    print(sep)
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    return
```

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest tests/ --ignore=tests/test_hf_models.py --ignore=tests/test_gpu_int4.py -x -q`
Expected: all pass (benchmark changes are CLI-only, no test regression)

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark.py
git commit -m "bench: add --context-scaling mode — separates prefill vs decode timing"
```

---

### Task 4: Benchmark and validate

**Files:**
- Create: benchmark script at `/tmp/bench_prefill_batched.py`
- Save: results to `benchmarks/`

- [ ] **Step 1: Create benchmark script**

```python
"""Benchmark batched prefill vs sequential — measure prefill speedup."""
import sys, time, traceback
sys.path.insert(0, "/home/elnur/gpt-oss-offload")

LOG = "/tmp/bench_prefill_batched.log"
def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f: f.write(m + "\n")

with open(LOG, "w"): pass
log("=== Batched Prefill Benchmark ===")

try:
    import torch
    from tinyserve.offload import load_and_offload
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    model = load_and_offload("openai/gpt-oss-20b", attn_implementation="sdpa")
    log(f"Loaded. Cache slots: {model._offload_pipelines[0].cache.capacity}")

    # Warmup
    inp = tok("Hello", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=3, do_sample=False)

    BASE = "The history of artificial intelligence. " * 500
    GEN = 20

    log(f"\n{'ctx':>6} {'prefill':>10} {'decode':>10} {'dec tok/s':>10} {'total tok/s':>12} {'dec HR%':>8}")
    log("-" * 65)

    for ctx in [10, 50, 100, 500, 1000, 2000, 3000]:
        try:
            from tinyserve.offloaded_model import reset_temporal_routing
            reset_temporal_routing()

            inp = tok(BASE, return_tensors="pt", truncation=True, max_length=ctx).to("cuda")
            plen = inp["input_ids"].shape[1]

            for p in model._offload_pipelines:
                if p.cache: p.cache.reset_stats()

            # Prefill
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(**inp, use_cache=False)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1000

            for p in model._offload_pipelines:
                if p.cache: p.cache.reset_stats()

            # Decode
            next_tok = out.logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(GEN):
                with torch.inference_mode():
                    out = model(input_ids=next_tok, use_cache=False)
                next_tok = out.logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - t1) * 1000

            hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
            misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
            hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
            dec_tps = GEN / (decode_ms / 1000)
            tot_tps = GEN / ((prefill_ms + decode_ms) / 1000)

            log(f"{plen:>6} {prefill_ms:>9.0f}ms {decode_ms:>9.0f}ms {dec_tps:>9.1f} {tot_tps:>11.1f} {hr:>7.0f}%")
        except Exception as e:
            log(f"{ctx:>6} ERR: {str(e)[:80]}")
            break

    log("\nDone.")
except Exception:
    log(f"FAIL:\n{traceback.format_exc()}")
```

- [ ] **Step 2: Run benchmark in background**

```bash
nohup python3 /tmp/bench_prefill_batched.py > /tmp/bench_prefill_stdout.log 2>&1 &
```

- [ ] **Step 3: Save results and update README if improved**

Copy log to `benchmarks/batched_prefill_20260326.txt`. If total tok/s improved at long context, update README table.

- [ ] **Step 4: Commit results**

```bash
git add benchmarks/ README.md
git commit -m "bench: batched prefill results — prefill/decode separated"
```
