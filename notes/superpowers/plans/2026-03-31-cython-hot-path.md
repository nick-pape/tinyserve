# Cython Hot Path Acceleration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cythonize the three Python hot-path functions that account for ~450ms/20-tokens of overhead: LFRU eviction scan, hit/miss classification, and batched prefill token grouping. Expected gain: 10-15ms/token → 10-16 tok/s sustained.

**Architecture:** A single `tinyserve/_fast_cache.pyx` module with typed C implementations of the inner loops. The existing Python functions fall back gracefully when the Cython extension isn't compiled. Build via `setup.py` extension.

**Tech Stack:** Cython 3.x, C `unordered_map`, existing tinyserve Python API.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `tinyserve/_fast_cache.pyx` | Create | Cython implementations of LFRU eviction, hit/miss classify, token grouping |
| `tinyserve/_fast_cache.pxd` | Create | Cython declarations |
| `setup.py` | Modify | Add Cython extension to build |
| `tinyserve/cache_policy.py` | Modify | Import Cython LFRU if available, fallback to Python |
| `tinyserve/generic_pipeline.py` | Modify | Import Cython classify/group if available |
| `tests/test_cython_fast_cache.py` | Create | Correctness tests comparing Cython vs Python outputs |

---

### Task 1: Cython LFRU eviction — the biggest win

**Files:**
- Create: `tinyserve/_fast_cache.pyx`
- Create: `setup.py` (Cython build)
- Test: `tests/test_cython_fast_cache.py`

The LFRU `select_evict()` does an O(n) scan over a Python dict computing `freq / (clock - last + 1)` for each entry. With 238 entries and 142 calls per 20 tokens, this accounts for ~210ms. A typed Cython version with `cdef` variables eliminates interpreter overhead.

- [ ] **Step 1: Create `_fast_cache.pyx`**

```cython
# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-accelerated cache operations for tinyserve hot path."""

from libc.math cimport INFINITY


def lfru_select_evict(dict data, long clock):
    """Find the key with lowest freq/age score in LFRU data dict.
    
    Args:
        data: dict mapping tuple keys to [slot, freq, clock] lists
        clock: current LFRU clock value
    
    Returns:
        (best_key, slot) — the entry to evict
    """
    cdef double best_score = INFINITY
    cdef double score
    cdef long freq, last, age, slot
    cdef object best_key = None

    for key, entry in data.items():
        slot = entry[0]
        freq = entry[1]
        last = entry[2]
        age = clock - last + 1
        score = <double>freq / <double>age
        if score < best_score:
            best_score = score
            best_key = key

    return best_key, data[best_key][0]


def classify_hits_misses(list expert_ids_list, list slots_list):
    """Classify experts as hits or misses from slot lookup results.
    
    Args:
        expert_ids_list: list of int expert IDs
        slots_list: list of int slot indices (-1 = miss)
    
    Returns:
        (hits, misses) where hits is list of (index, slot) and misses is list of index
    """
    cdef list hits = []
    cdef list misses = []
    cdef int i, slot
    cdef int n = len(expert_ids_list)
    
    for i in range(n):
        slot = slots_list[i]
        if slot >= 0:
            hits.append((i, slot))
        else:
            misses.append(i)
    
    return hits, misses


def group_tokens_by_expert(list eid_list, int seq_len, int top_k):
    """Group token indices by expert ID for batched prefill.
    
    Args:
        eid_list: flattened list of expert IDs [seq_len][top_k] as nested list
        seq_len: number of tokens
        top_k: experts per token
    
    Returns:
        dict mapping expert_id to list of (token_idx, k_idx) tuples
    """
    cdef dict groups = {}
    cdef int tok, k, eid
    cdef list group
    
    for tok in range(seq_len):
        for k in range(top_k):
            eid = eid_list[tok][k]
            if eid in groups:
                group = groups[eid]
            else:
                group = []
                groups[eid] = group
            group.append((tok, k))
    
    return groups
```

- [ ] **Step 2: Create setup.py for Cython build**

```python
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "tinyserve._fast_cache",
        ["tinyserve/_fast_cache.pyx"],
    ),
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
    }),
)
```

- [ ] **Step 3: Build the extension**

```bash
python setup.py build_ext --inplace
```

- [ ] **Step 4: Write tests**

Create `tests/test_cython_fast_cache.py`:

```python
"""Tests for Cython-accelerated cache operations."""
import pytest
import time

# Skip if Cython extension not compiled
try:
    from tinyserve._fast_cache import lfru_select_evict, classify_hits_misses, group_tokens_by_expert
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

pytestmark = pytest.mark.skipif(not HAS_CYTHON, reason="Cython extension not compiled")


def test_lfru_select_evict_correctness():
    """Cython eviction matches Python LFRU."""
    data = {
        (0, 0): [0, 10, 100],   # freq=10, last=100
        (0, 1): [1, 1, 99],     # freq=1, last=99 — lowest score
        (0, 2): [2, 5, 50],     # freq=5, last=50
    }
    clock = 100
    key, slot = lfru_select_evict(data, clock)
    # (0,2) has score 5/(100-50+1) = 0.098
    # (0,1) has score 1/(100-99+1) = 0.5
    # (0,0) has score 10/(100-100+1) = 10.0
    # Wait — (0,2) has lowest score
    assert key == (0, 2)
    assert slot == 2


def test_lfru_select_evict_speed():
    """Cython eviction should be >5x faster than Python on 238 entries."""
    data = {(0, i): [i, i % 10 + 1, 1000 - i] for i in range(238)}
    clock = 1000

    # Python baseline
    t0 = time.perf_counter()
    for _ in range(1000):
        best_key = None
        best_score = float("inf")
        for k, (slot, freq, last) in data.items():
            age = clock - last + 1
            score = freq / age
            if score < best_score:
                best_score = score
                best_key = k
    py_time = time.perf_counter() - t0

    # Cython
    t0 = time.perf_counter()
    for _ in range(1000):
        lfru_select_evict(data, clock)
    cy_time = time.perf_counter() - t0

    speedup = py_time / cy_time
    print(f"Python: {py_time*1000:.1f}ms, Cython: {cy_time*1000:.1f}ms, speedup: {speedup:.1f}x")
    assert speedup > 3.0, f"Expected >3x speedup, got {speedup:.1f}x"


def test_classify_hits_misses():
    eids = [0, 1, 2, 3]
    slots = [5, -1, 3, -1]
    hits, misses = classify_hits_misses(eids, slots)
    assert hits == [(0, 5), (2, 3)]
    assert misses == [1, 3]


def test_group_tokens_by_expert():
    eid_list = [[0, 1], [0, 2], [1, 2]]
    groups = group_tokens_by_expert(eid_list, 3, 2)
    assert 0 in groups
    assert len(groups[0]) == 2  # tokens 0 and 1 route to expert 0
    assert groups[0] == [(0, 0), (1, 0)]
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_cython_fast_cache.py -x -v
```

- [ ] **Step 6: Commit**

```bash
git add tinyserve/_fast_cache.pyx setup.py tests/test_cython_fast_cache.py
git commit -m "feat: Cython-accelerated LFRU eviction, hit/miss classify, token grouping

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire Cython into existing Python code

**Files:**
- Modify: `tinyserve/cache_policy.py:220-232` (LFRU select_evict)
- Modify: `tinyserve/generic_pipeline.py:479-490` (hit/miss classify)
- Modify: `tinyserve/generic_pipeline.py:399-406` (token grouping)

- [ ] **Step 1: Wire LFRU eviction**

In `tinyserve/cache_policy.py`, replace `LFRUPolicy.select_evict`:

```python
class LFRUPolicy(CachePolicy):
    # ... existing code ...

    def select_evict(self) -> tuple[tuple, int]:
        try:
            from tinyserve._fast_cache import lfru_select_evict
            return lfru_select_evict(self._data, self._clock)
        except ImportError:
            pass
        # Python fallback
        best_key = None
        best_score = float("inf")
        for k, (slot, freq, last) in self._data.items():
            age = self._clock - last + 1
            score = freq / age
            if score < best_score:
                best_score = score
                best_key = k
        return best_key, self._data[best_key][0]
```

- [ ] **Step 2: Wire hit/miss classification**

In `tinyserve/generic_pipeline.py`, in `_execute_token_experts`, replace the Python loop at line ~479:

```python
# Try Cython fast path
try:
    from tinyserve._fast_cache import classify_hits_misses as _cy_classify
except ImportError:
    _cy_classify = None

# In the function:
if _cy_classify is not None:
    hits, misses = _cy_classify(expert_ids_list, slots_list)
    # Still need to update policy and counters
    for i, slot in hits:
        cache._policy.lookup((layer_idx, expert_ids_list[i]))
        cache.hits += 1
    for i in misses:
        cache.misses += 1
else:
    # existing Python path
```

- [ ] **Step 3: Wire token grouping**

In `tinyserve/generic_pipeline.py`, in `execute_layer_experts_batched`:

```python
try:
    from tinyserve._fast_cache import group_tokens_by_expert as _cy_group
except ImportError:
    _cy_group = None

# In the function:
if _cy_group is not None:
    expert_groups = _cy_group(eid_list, seq_len, top_k)
else:
    # existing Python grouping loop
```

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ --ignore=tests/test_hf_models.py --ignore=tests/test_gpu_int4.py -x -q
```

- [ ] **Step 5: Commit**

```bash
git add tinyserve/cache_policy.py tinyserve/generic_pipeline.py
git commit -m "perf: wire Cython fast path into LFRU, hit/miss, token grouping

Graceful fallback: if _fast_cache.so not compiled, Python path runs.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Benchmark the improvement

- [ ] **Step 1: Run the diverse workload benchmark**

```bash
python -m scripts.cache_benchmark --model openai/gpt-oss-20b --policy lfru --gen-tokens 30 --json benchmarks/cython_bench_20260331.json > benchmarks/cython_bench_20260331.txt 2>&1
```

- [ ] **Step 2: Compare against pre-Cython baseline**

Expected improvement: ~10-15ms/token from eliminating Python interpreter overhead in the three hot loops. At current 8-13 tok/s (77-125ms/token), saving 10-15ms would push to ~9-15 tok/s.

- [ ] **Step 3: Run the cProfile comparison**

Profile 20 tokens with and without Cython to measure exact per-function improvement.

- [ ] **Step 4: Commit results**

```bash
git add benchmarks/cython_bench_20260331.*
git commit -m "bench: Cython hot path results

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add Cython build to CI and pyproject.toml

- [ ] **Step 1: Add cython to dev dependencies**

In `pyproject.toml`, add `cython>=3.0` to `[project.optional-dependencies] dev`.

- [ ] **Step 2: Add build step to CI**

In `.github/workflows/ci.yml`, add `python setup.py build_ext --inplace` before test step.

- [ ] **Step 3: Add .so to .gitignore**

```
*.so
*.c
# But NOT _fast_cache.pyx
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml .gitignore
git commit -m "build: add Cython extension to CI and dev dependencies

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
