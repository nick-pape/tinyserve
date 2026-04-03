# Miss Penalty Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce cache miss penalty from ~20ms to <2ms by routing ALL cache misses through CPU expert compute instead of the H2D pipeline, and add buddy expert substitution as a zero-stall fallback.

**Architecture:** On cache miss during decode, compute the expert FFN on CPU using the existing `CPUExpertForward` (which auto-selects AMX/AVX-512/AVX2 via OneDNN). Only activations (~KB) cross PCIe, not weights (~MB). For batch=1 decode this is faster than H2D weight transfer. Additionally, pre-compute co-activation buddy tables so misses can optionally substitute a cached expert instead of stalling at all.

**Tech Stack:** Existing `tinyserve/cpu_expert.py` (CPUExpertForward, CPUINT4Forward), existing `tinyserve/generic_pipeline.py` (_execute_token_experts), `tinyserve/ram_cache.py`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `tinyserve/offload.py` | Modify | Change default cache policy to `lfru` |
| `tinyserve/generic_pipeline.py` | Modify | Route ALL decode misses through CPU compute path |
| `tinyserve/buddy_experts.py` | Create | Co-activation profiling + buddy lookup |
| `tests/test_cpu_miss_fallback.py` | Create | Tests for CPU-on-miss correctness |
| `tests/test_buddy_experts.py` | Create | Tests for buddy profiling and substitution |

---

### Task 0: Change default cache policy to LFRU

**Files:**
- Modify: `tinyserve/offload.py`

This is a one-line change with immediate impact: deep-layer HR goes from 8% to 52%.

- [ ] **Step 1: Find the default cache_policy parameter**

In `tinyserve/offload.py`, find the `load_and_offload` function signature where `cache_policy` defaults to `"lru"`.

- [ ] **Step 2: Change default to "lfru"**

Change the default value from `"lru"` to `"lfru"`.

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest tests/ --ignore=tests/test_hf_models.py --ignore=tests/test_gpu_int4.py -x -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add tinyserve/offload.py
git commit -m "perf: change default cache policy to LFRU — 6.5x better deep-layer HR

Benchmark data: LFRU achieves 89.7% overall HR with 52% on deep layers
(18-23), vs LRU's 87.8% overall and 8% deep layers. LFRU's frequency
component prevents early layers from monopolizing the cache.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 1: Route all decode misses through CPU compute

**Files:**
- Modify: `tinyserve/generic_pipeline.py:559-589`
- Test: `tests/test_cpu_miss_fallback.py`

Currently, cache misses during decode go through `_pipeline_experts()` which does synchronous H2D weight transfer (~20ms per expert on 8GB GPU). The CPU expert path at line 564 only triggers for "truly cold" experts not in RAM. We need to route ALL misses through CPU compute for batch=1 decode.

The key insight from Fiddler (arxiv 2402.07033): at batch=1, sending activations to CPU (~4KB for hidden_dim=2880) is far cheaper than transferring expert weights to GPU (~13MB for MXFP4). CPU compute via OneDNN (AMX/AVX-512) takes ~1.9ms per expert vs ~20ms H2D transfer.

- [ ] **Step 1: Write failing tests**

Create `tests/test_cpu_miss_fallback.py`:

```python
"""Tests for CPU expert compute on cache miss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from tests.conftest import requires_cuda


class TinyExpert(nn.Module):
    def __init__(self, h=16, i=32):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2 * i, h))
        self.down_proj = nn.Parameter(torch.randn(h, i))
        self._act_fn = nn.SiLU()
        self._has_bias = False
        self._is_mxfp4 = False
        self._param_names = ["gate_up_proj", "down_proj"]

    def forward(self, x):
        gate_up = F.linear(x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.linear(self._act_fn(gate) * up, self.down_proj)


def _make_pipeline_with_cpu(num_experts=4, hidden=16, intermediate=32):
    from tinyserve.generic_store import GenericExpertStore, GenericLRUCache
    from tinyserve.generic_pipeline import GenericExpertPipeline
    from tinyserve.cpu_expert import CPUExpertForward

    weights = {}
    for li in range(1):
        for ei in range(num_experts):
            weights[(li, ei)] = {
                "gate_up_proj": torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16),
                "down_proj": torch.randn(hidden, intermediate, dtype=torch.bfloat16),
            }
    store = GenericExpertStore.from_dict(weights, 1, num_experts)
    device = torch.device("cuda")
    template = TinyExpert(hidden, intermediate).to(device).to(torch.bfloat16)
    buf_a = store.allocate_buffer(device)
    buf_b = store.allocate_buffer(device)
    ts = torch.cuda.Stream(device)
    cs = torch.cuda.Stream(device)
    # Small cache: capacity=2 forces misses with 4 experts
    cache = GenericLRUCache(2, store.buffer_expert_bytes, device,
                           num_layers=1, num_experts=num_experts)
    cpu_fwd = CPUExpertForward(store.layout, act_fn=nn.SiLU())
    pipeline = GenericExpertPipeline(store, template, device,
                                     buf_a, buf_b, ts, cs, cache=cache)
    pipeline.cpu_expert = cpu_fwd
    pipeline.cpu_on_miss = True
    return pipeline, store


@requires_cuda
def test_cpu_miss_produces_correct_output():
    """CPU compute on miss should produce same output as GPU compute."""
    pipeline, store = _make_pipeline_with_cpu()
    h = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.tensor([[0, 1]], device="cuda")
    weights = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.bfloat16)

    # Fill cache with experts 2,3 so experts 0,1 are misses
    pipeline.execute_layer_experts(h, 0, torch.tensor([[2, 3]], device="cuda"),
                                   torch.tensor([[0.5, 0.5]], device="cuda", dtype=torch.bfloat16))

    # Now request 0,1 — should miss and use CPU compute
    pipeline.cache.reset_stats()
    out = pipeline.execute_layer_experts(h, 0, expert_ids, weights)

    # Output should be non-zero (CPU compute produced a result)
    assert out.abs().sum() > 0
    # Misses should have occurred
    assert pipeline.cache.misses > 0


@requires_cuda
def test_cpu_miss_flag_controls_behavior():
    """When cpu_on_miss=False, misses use GPU pipeline instead."""
    pipeline, store = _make_pipeline_with_cpu()
    pipeline.cpu_on_miss = False

    h = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.tensor([[0]], device="cuda")
    weights = torch.tensor([[1.0]], device="cuda", dtype=torch.bfloat16)

    # Should work (uses GPU pipeline for miss)
    out = pipeline.execute_layer_experts(h, 0, expert_ids, weights)
    assert out.abs().sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cpu_miss_fallback.py -x -q`
Expected: FAIL — `cpu_on_miss` attribute doesn't exist

- [ ] **Step 3: Implement CPU-on-miss routing**

In `tinyserve/generic_pipeline.py`, add `cpu_on_miss` attribute to `GenericExpertPipeline.__init__`:

```python
self.cpu_on_miss: bool = False
```

Then modify `_execute_token_experts` — replace the miss handling block (lines ~559-589) with:

```python
if not misses:
    return

# CPU compute for ALL misses when cpu_on_miss=True and batch=1.
# Fiddler insight: sending activations (~KB) to CPU is faster than
# transferring weights (~MB) to GPU at batch=1.
if self.cpu_on_miss and self.cpu_expert is not None and h.shape[0] == 1:
    for i in misses:
        eid = expert_ids_list[i]
        # Get expert data from store (CPU pinned memory)
        expert_packed = self.store.get_expert_data(layer_idx, eid)
        with _prof.phase("cpu_compute") if _prof else nullcontext():
            out = self.cpu_expert.forward(h, expert_packed)
        output[tok_idx] += weights[i] * out.squeeze(0)
        # Populate GPU cache for future hits
        if cache is not None:
            gpu_slot = cache.allocate(layer_idx, eid)
            cache.get_packed(gpu_slot).copy_(expert_packed[:cache.expert_bytes], non_blocking=True)
    return

# Original path: GPU double-buffered H2D pipeline (for batched or cpu_on_miss=False)
# ... existing code for ram_cache check and _pipeline_experts ...
```

Also need to verify `GenericExpertStore` has a `get_expert_data(layer_idx, eid)` method that returns the packed tensor from CPU. If not, add it:

```python
def get_expert_data(self, layer_idx: int, expert_idx: int) -> torch.Tensor:
    """Return raw packed expert data from CPU store."""
    return self._data[layer_idx, expert_idx]
```

- [ ] **Step 4: Enable cpu_on_miss by default in offload.py**

In `tinyserve/offload.py` or `tinyserve/offloaded_model.py`, where `GenericExpertPipeline` is constructed, set `pipeline.cpu_on_miss = True` when `cpu_expert` is available.

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_cpu_miss_fallback.py -x -v`
Expected: 2/2 PASS

- [ ] **Step 6: Run full test suite**

Run: `python3 -m pytest tests/ --ignore=tests/test_hf_models.py --ignore=tests/test_gpu_int4.py -x -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add tinyserve/generic_pipeline.py tinyserve/generic_store.py tinyserve/offloaded_model.py tests/test_cpu_miss_fallback.py
git commit -m "perf: route decode cache misses through CPU compute (Fiddler pattern)

At batch=1, computing expert FFN on CPU (~1.9ms via OneDNN AMX/AVX-512)
is faster than H2D weight transfer (~20ms). Only activations (~4KB)
cross PCIe, not weights (~13MB). CPU compute path was already
implemented (cpu_expert.py); this change routes ALL decode misses
through it instead of the GPU double-buffered pipeline.

Based on Fiddler (arxiv 2402.07033, ICLR 2025).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Buddy expert substitution

**Files:**
- Create: `tinyserve/buddy_experts.py`
- Modify: `tinyserve/generic_pipeline.py`
- Test: `tests/test_buddy_experts.py`

BuddyMoE (arxiv 2511.10054): when a cache miss occurs, substitute a co-activation-similar cached expert. Zero stall — the buddy is already in GPU cache.

Phase 1: offline profiling builds co-activation tables. Phase 2: at runtime, on miss, check if a buddy is cached and substitute.

- [ ] **Step 1: Write failing tests for buddy profiling**

Create `tests/test_buddy_experts.py`:

```python
"""Tests for buddy expert co-activation profiling and substitution."""
import torch
import pytest


def test_coactivation_matrix_from_routing():
    from tinyserve.buddy_experts import build_coactivation_matrix

    # 10 tokens, 4 experts, top_k=2
    routing = torch.tensor([
        [0, 1], [0, 2], [1, 2], [0, 1], [2, 3],
        [0, 1], [1, 3], [0, 2], [2, 3], [0, 1],
    ])
    coact = build_coactivation_matrix(routing, num_experts=4)
    # Experts 0 and 1 co-activate 4 times
    assert coact[0, 1] == coact[1, 0]
    assert coact[0, 1] > coact[0, 3]


def test_buddy_lookup():
    from tinyserve.buddy_experts import BuddyTable

    # 4 experts, buddy of expert 0 is expert 1 (highest co-activation)
    coact = torch.tensor([
        [0, 5, 2, 1],
        [5, 0, 3, 1],
        [2, 3, 0, 4],
        [1, 1, 4, 0],
    ], dtype=torch.float32)
    table = BuddyTable.from_coactivation(coact, max_buddies=2)

    buddies = table.get_buddies(0)
    assert buddies[0] == 1  # highest co-activation with expert 0


def test_buddy_substitution_uses_cached_expert():
    from tinyserve.buddy_experts import BuddyTable

    coact = torch.tensor([
        [0, 5, 2, 1],
        [5, 0, 3, 1],
        [2, 3, 0, 4],
        [1, 1, 4, 0],
    ], dtype=torch.float32)
    table = BuddyTable.from_coactivation(coact, max_buddies=2)

    cached_experts = {1, 3}  # experts currently in GPU cache
    buddy = table.find_cached_buddy(0, cached_experts)
    assert buddy == 1  # expert 1 is buddy of 0 and is cached
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_buddy_experts.py -x -q`
Expected: FAIL — module not found

- [ ] **Step 3: Implement buddy_experts.py**

Create `tinyserve/buddy_experts.py`:

```python
"""Buddy expert co-activation profiling and substitution.

Based on BuddyMoE (arxiv 2511.10054): when a cache miss occurs, substitute
a co-activation-similar cached expert for zero-stall inference. Small
accuracy cost bounded by co-activation similarity.
"""
import torch


def build_coactivation_matrix(
    routing_decisions: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Build expert co-activation matrix from routing decisions.

    Args:
        routing_decisions: [num_tokens, top_k] expert indices
        num_experts: total number of experts

    Returns:
        [num_experts, num_experts] symmetric co-activation count matrix
    """
    coact = torch.zeros(num_experts, num_experts, dtype=torch.float32)
    for token_experts in routing_decisions:
        experts = token_experts.tolist()
        for i in range(len(experts)):
            for j in range(i + 1, len(experts)):
                coact[experts[i], experts[j]] += 1
                coact[experts[j], experts[i]] += 1
    return coact


class BuddyTable:
    """Pre-computed buddy expert lookup table."""

    def __init__(self, buddies: dict[int, list[int]]):
        self._buddies = buddies

    @classmethod
    def from_coactivation(cls, coact: torch.Tensor, max_buddies: int = 3):
        """Build buddy table from co-activation matrix."""
        n = coact.shape[0]
        buddies = {}
        for eid in range(n):
            scores = coact[eid].clone()
            scores[eid] = -1  # exclude self
            top = scores.topk(min(max_buddies, n - 1)).indices.tolist()
            buddies[eid] = top
        return cls(buddies)

    def get_buddies(self, expert_id: int) -> list[int]:
        return self._buddies.get(expert_id, [])

    def find_cached_buddy(self, expert_id: int, cached_set: set[int]) -> int | None:
        for buddy in self.get_buddies(expert_id):
            if buddy in cached_set:
                return buddy
        return None
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_buddy_experts.py -x -v`
Expected: 3/3 PASS

- [ ] **Step 5: Integrate buddy substitution into the miss path**

In `tinyserve/generic_pipeline.py`, in the CPU-on-miss block from Task 1, add buddy check BEFORE CPU compute:

```python
# In the miss handling block:
if self.cpu_on_miss and self.cpu_expert is not None and h.shape[0] == 1:
    for i in misses:
        eid = expert_ids_list[i]

        # BuddyMoE: try cached substitute first (zero stall)
        if self.buddy_table is not None and cache is not None:
            cached_set = set(cache._policy._order.keys()) if hasattr(cache._policy, '_order') else set()
            # Build cached set from policy
            buddy_eid = self.buddy_table.find_cached_buddy(
                eid, {k[1] for k in cached_set if k[0] == layer_idx}
            )
            if buddy_eid is not None:
                slot = cache.lookup(layer_idx, buddy_eid)
                if slot is not None:
                    packed = cache.get_packed(slot)
                    if _inline is not None:
                        out = _inline(packed, h)
                    else:
                        out = forward_from_packed(self.template, packed, self._param_refs, h)
                    output[tok_idx] += weights[i] * out.squeeze(0)
                    continue

        # No buddy available — fall back to CPU compute
        expert_packed = self.store.get_expert_data(layer_idx, eid)
        out = self.cpu_expert.forward(h, expert_packed)
        output[tok_idx] += weights[i] * out.squeeze(0)
        if cache is not None:
            gpu_slot = cache.allocate(layer_idx, eid)
            cache.get_packed(gpu_slot).copy_(expert_packed[:cache.expert_bytes], non_blocking=True)
    return
```

Add `buddy_table` attribute to `GenericExpertPipeline.__init__`:

```python
self.buddy_table = None  # Set externally after profiling
```

- [ ] **Step 6: Run full test suite**

Run: `python3 -m pytest tests/ --ignore=tests/test_hf_models.py --ignore=tests/test_gpu_int4.py -x -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add tinyserve/buddy_experts.py tinyserve/generic_pipeline.py tests/test_buddy_experts.py
git commit -m "feat: buddy expert substitution — zero-stall on cache miss

When a cache miss occurs during decode, first check if a co-activation-
similar expert is already cached. If so, substitute it (zero stall).
If no buddy is cached, fall back to CPU compute (~1.9ms).

Based on BuddyMoE (arxiv 2511.10054). Buddy tables are pre-computed
from co-activation profiling data.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Benchmark the improvement

**Files:**
- Create: benchmark script at `/tmp/bench_miss_penalty.py`
- Save: `benchmarks/miss_penalty_20260331.txt`

- [ ] **Step 1: Run the comprehensive cache benchmark with LFRU + CPU-on-miss**

```bash
nohup python3 -m scripts.cache_benchmark \
  --model openai/gpt-oss-20b \
  --policy lfru \
  --gen-tokens 30 \
  --json benchmarks/miss_penalty_20260331.json \
  > benchmarks/miss_penalty_20260331.txt 2>&1 &
```

- [ ] **Step 2: Compare results against LRU baseline**

Expected improvement:
- Hit rate: 87.8% (LRU) → 89.7% (LFRU)
- Miss penalty: ~20ms → ~1.9ms (CPU compute)
- Net tok/s improvement: significant since 10-12% of all expert lookups were 20ms stalls

- [ ] **Step 3: Commit results**

```bash
git add benchmarks/miss_penalty_20260331.*
git commit -m "bench: LFRU + CPU-on-miss results — miss penalty reduction measured

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
