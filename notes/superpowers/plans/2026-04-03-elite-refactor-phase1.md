# Elite Refactor Phase 1 — Structural Renames + API Cleanup

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform tinyserve from "working research prototype" to "a principal engineer says 'well-built' in 30 seconds." Phase 1 covers structural renames, dead code pruning, and API surface cleanup — all independent tasks with no behavioral changes.

**Architecture:** Mechanical find/replace for renames across ~25 files, plus targeted new code for API improvements (RoutingSpec, AttentionBackend, TinyserveConfig). Every task must leave all tests passing. Backward-compat shims for renamed modules.

**Tech Stack:** Python 3.11+, PyTorch, existing tinyserve infrastructure.

**Spec:** `docs/superpowers/specs/2026-04-03-elite-refactor-design.md`

---

## Execution Strategy

Phase 1 has 11 independent tasks. To avoid merge conflicts on parallel agents touching the same files, **execute sequentially in this order** (tasks that touch many files first, isolated tasks last):

1. Task 1: Rename generic_* → expert_* (touches ~25 files — do first)
2. Task 2: Rename offloaded_model.py → _model_hooks.py (touches ~8 files)
3. Task 3: RoutingSpec namedtuple (touches offload.py only)
4. Task 4: AttentionBackend enum (touches offload.py only)
5. Task 7: logger.warning() on silent fallbacks (touches ~6 files)
6. Task 8: Dead code removal (touches ~3 files)
7. Task 5: Rename buf_a/buf_b (touches expert_pipeline.py only)
8. Task 6: Rename _cy_classify etc. (touches expert_pipeline.py only)
9. Task 9: Prune scripts/ (delete 16 files)
10. Task 10: Move internal docs (move 4 files)
11. Task 11: Trim __init__.py exports

**Commit after each task. Run tests after each task.**

---

### Task 1: Rename generic_* → expert_*

**Files:**
- Rename: `tinyserve/generic_pipeline.py` → `tinyserve/expert_pipeline.py`
- Rename: `tinyserve/generic_store.py` → `tinyserve/expert_store.py`
- Rename: `tests/test_generic_pipeline.py` → `tests/test_expert_pipeline.py`
- Rename: `tests/test_generic_store.py` → `tests/test_expert_store.py`
- Create: `tinyserve/generic_pipeline.py` (deprecation shim, 8 lines)
- Create: `tinyserve/generic_store.py` (deprecation shim, 8 lines)
- Modify: ~25 files (import updates)

- [ ] **Step 1: Rename the files**

```bash
git mv tinyserve/generic_pipeline.py tinyserve/expert_pipeline.py
git mv tinyserve/generic_store.py tinyserve/expert_store.py
git mv tests/test_generic_pipeline.py tests/test_expert_pipeline.py
git mv tests/test_generic_store.py tests/test_expert_store.py
```

- [ ] **Step 2: Find and replace all class names in the renamed files**

In `tinyserve/expert_pipeline.py`:
- `GenericExpertPipeline` → `ExpertPipeline`

In `tinyserve/expert_store.py`:
- `GenericExpertStore` → `ExpertStore`
- `GenericExpertBuffer` → `ExpertBuffer`
- `GenericLRUCache` → `ExpertCache`

Use `replace_all=True` for each substitution.

- [ ] **Step 3: Update all imports across the codebase**

For every file listed in the spec (Task 1, "All files requiring import updates"), replace:
- `from .generic_store import` → `from .expert_store import`
- `from .generic_pipeline import` → `from .expert_pipeline import`
- `from tinyserve.generic_store import` → `from tinyserve.expert_store import`
- `from tinyserve.generic_pipeline import` → `from tinyserve.expert_pipeline import`
- `GenericExpertPipeline` → `ExpertPipeline`
- `GenericExpertStore` → `ExpertStore`
- `GenericExpertBuffer` → `ExpertBuffer`
- `GenericLRUCache` → `ExpertCache`

Files to update (from spec): expert_pipeline.py, offloaded_model.py, offload.py, expert_batcher.py, gpu_int4.py, gguf_loader.py, gguf_store.py, imatrix.py, __init__.py, + all test files that import these classes, + scripts/benchmark.py.

- [ ] **Step 4: Create backward-compat shims**

Create `tinyserve/generic_pipeline.py`:
```python
"""Deprecated: use tinyserve.expert_pipeline instead."""
import warnings
warnings.warn(
    "tinyserve.generic_pipeline is deprecated, use tinyserve.expert_pipeline",
    DeprecationWarning, stacklevel=2,
)
from tinyserve.expert_pipeline import *  # noqa: F401,F403
```

Create `tinyserve/generic_store.py`:
```python
"""Deprecated: use tinyserve.expert_store instead."""
import warnings
warnings.warn(
    "tinyserve.generic_store is deprecated, use tinyserve.expert_store",
    DeprecationWarning, stacklevel=2,
)
from tinyserve.expert_store import *  # noqa: F401,F403
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/ --ignore=tests/test_hf_models.py --ignore=tests/test_gpu_int4.py -x -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rename generic_* → expert_* — files, classes, imports

GenericExpertPipeline → ExpertPipeline
GenericExpertStore → ExpertStore
GenericExpertBuffer → ExpertBuffer
GenericLRUCache → ExpertCache

Backward-compat shims with DeprecationWarning for external importers.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Rename offloaded_model.py → _model_hooks.py

**Files:**
- Rename: `tinyserve/offloaded_model.py` → `tinyserve/_model_hooks.py`
- Rename: `tests/test_offloaded_model.py` → `tests/test_model_hooks.py`
- Create: `tinyserve/offloaded_model.py` (deprecation shim)
- Modify: offload.py, expert_pipeline.py, + test/script files

- [ ] **Step 1: Rename files**
- [ ] **Step 2: Update all imports** (see spec Task 2 for exhaustive list)
- [ ] **Step 3: Create backward-compat shim**
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: offloaded_model.py → _model_hooks.py (internal module)"
```

---

### Task 3: RoutingSpec namedtuple

**Files:**
- Modify: `tinyserve/offload.py`

- [ ] **Step 1: Replace raw tuples with RoutingSpec**

See spec Task 3 for the exact code. Add `RoutingSpec` as a `NamedTuple` with `softmax_order`, `returns_logits`, `router_attr` fields. Update all `_ROUTING_MAP` entries. Update the unpacking site (~line 291).

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: RoutingSpec namedtuple replaces undocumented tuples in _ROUTING_MAP"
```

---

### Task 4: AttentionBackend enum

**Files:**
- Modify: `tinyserve/offload.py`

- [ ] **Step 1: Create enum and replace magic strings**

See spec Task 4. Add `class AttentionBackend(str, Enum)` with EAGER, SDPA, FLEX, FLASHINFER, FLASH_ATTENTION_2. Update `attn_implementation` parameter type and all string comparisons.

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: AttentionBackend enum replaces magic strings"
```

---

### Task 5: Rename buf_a/buf_b → staging_buffer_a/staging_buffer_b

**Files:**
- Modify: `tinyserve/expert_pipeline.py` (after Task 1 rename)

- [ ] **Step 1: Replace all buf_a/buf_b references**

`self.buf_a` → `self.staging_buffer_a`, `self.buf_b` → `self.staging_buffer_b`. Use `replace_all=True`.

Also update `offloaded_model.py` / `_model_hooks.py` where `buf_a`/`buf_b` are constructed.

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: buf_a/buf_b → staging_buffer_a/b (descriptive naming)"
```

---

### Task 6: Rename Cython abbreviations

**Files:**
- Modify: `tinyserve/expert_pipeline.py`

- [ ] **Step 1: Rename**

`_cy_classify` → `_cython_classify_hits`, `_cy_group` → `_cython_group_by_expert`. Update all references.

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: _cy_classify → _cython_classify_hits (no abbreviations)"
```

---

### Task 7: Add logger.warning() to all silent fallbacks

**Files:**
- Modify: `tinyserve/offload.py`, `tinyserve/expert_pipeline.py`, `tinyserve/expert_store.py`

- [ ] **Step 1: Find all silent exception handlers**

Search for `except Exception`, `except ImportError`, `except OSError` and any auto-sized-to-zero patterns. See spec Task 7 for the ~17 sites.

- [ ] **Step 2: Add warnings**

Each silent fallback gets `logger.warning("description of what failed and what we're falling back to")`.

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: logger.warning() on all 17 silent fallback sites"
```

---

### Task 8: Dead code removal

**Files:**
- Modify: `tinyserve/expert_pipeline.py`, `tinyserve/cache_policy.py`, `tinyserve/mxfp4.py`

- [ ] **Step 1: Remove all #noqa suppressions**

Find each `# noqa` and either fix the underlying issue or delete the dead code.

- [ ] **Step 2: Remove unreachable code**

`_build_gpu_int4_forward` body after `return None`, unused `dequant_single_expert`, unused `estimate_capacity`.

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: remove dead code — noqa suppressions, unreachable paths, unused functions"
```

---

### Task 9: Prune scripts/

**Files:**
- Delete: 16 script files (see spec Task 9)
- Keep: `scripts/benchmark.py`, `scripts/cache_benchmark.py`, `scripts/prompts.py`, `scripts/__init__.py`

- [ ] **Step 1: Delete files**

```bash
git rm scripts/bench_attention.py scripts/bench_disk_offload.py scripts/bench_flex_only.py \
      scripts/bench_kv_configs.py scripts/bench_long_context.py scripts/bench_qwen35.py \
      scripts/bench_with_buddies.py scripts/comprehensive_bench.py scripts/debug_bench.py \
      scripts/sweep_cache_bias.py scripts/validate_context_scaling.py scripts/autotune.py \
      scripts/calibrate_buddies.py scripts/expert_similarity.py \
      scripts/test_dynamic_rebalance.py scripts/test_qwen_122b.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "refactor: prune scripts/ from 20 to 4 essential files

Benchmark data preserved in benchmarks/. One-off experiment scripts
served their purpose and are now documentation, not tools."
```

---

### Task 10: Move internal docs

**Files:**
- Move: `docs/gpt-oss-20b-architecture.md` → `notes/gpt-oss-20b-architecture.md`
- Move: `docs/superpowers/` → `notes/superpowers/` (planning docs)
- Keep: `docs/` empty or with user-facing docs only

- [ ] **Step 1: Create notes/ and move**

```bash
mkdir -p notes
git mv docs/gpt-oss-20b-architecture.md notes/
git mv docs/superpowers notes/superpowers
```

- [ ] **Step 2: Commit**

```bash
git commit -m "refactor: move internal planning docs to notes/ — docs/ is user-facing only"
```

---

### Task 11: Trim __init__.py exports

**Files:**
- Modify: `tinyserve/__init__.py`

- [ ] **Step 1: Read current exports and trim**

Keep only: `load_and_offload`, `offload_model`, `load_from_gguf`, `__version__`.
Remove: `chunked_prefill`, `generate_chunked`, `PagedKVPool`, `PagedRequestKVCache`, `StaticKVCache`, and any other internal exports.

Users needing internals can import from submodules: `from tinyserve.static_kv_cache import StaticKVCache`.

- [ ] **Step 2: Update any tests that import from tinyserve directly**
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: trim __init__.py to 3 public functions — internals via submodules"
```
