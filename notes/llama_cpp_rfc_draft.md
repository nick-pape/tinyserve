# RFC DRAFT — DO NOT POST AS-IS
# You must rewrite this in your own words before posting to GitHub.
# Posting AI-generated text to llama.cpp issues violates their CONTRIBUTING.md.
# This draft covers all the technical points you'll want to hit.
# ─────────────────────────────────────────────────────────────────────────────

---
**Issue title:** Feature Request: RAM-tier expert cache for MoE models (`--n-cpu-moe` + eviction policy)
**Labels:** enhancement, help wanted
**Repository:** https://github.com/ggml-org/llama.cpp
---

### Prerequisites

- [ ] I am running the latest code. Mention the version if possible as well.
- [ ] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [ ] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

---

### Feature Description

## Problem

Running a large MoE model like GPT-OSS-120B (128 experts/layer, 36 layers, ~57 GB of expert weights) on a GPU with 8 GB VRAM currently works via `--n-gpu-layers` with the MoE tensor pattern matching that keeps expert weights in CPU RAM. The GPU handles the dense/attention layers while expert weights are demand-paged from RAM on each forward pass.

The bottleneck is that every decode step re-reads the same ~4 hot experts per layer from RAM, copies them across PCIe, and discards them. For GPT-OSS-120B with top-4 routing across 36 layers, that's 144 RAM→GPU copies per token — most of them reading the same experts as the previous token.

MoE routing is not uniform. Empirically, a small set of experts (often ~10-20% of the total) handles the majority of tokens. A fixed-size RAM-side cache holding those hot experts in CPU pinned memory — sized to fit comfortably in system RAM alongside the mmap'd full weight tensor — would cut PCIe traffic dramatically during decode once the cache warms up.

## Proof of concept (Python, HuggingFace)

I built a Python PoC ([gpt-oss-offload](LINK_TO_YOUR_REPO)) that implements this three-tier approach on top of HuggingFace transformers for GPT-OSS-120B:

```
Tier 1: GPU VRAM  — small LRU cache of top-K expert slots  (fixed-address buffers)
Tier 2: CPU RAM   — pinned memory holding hot experts       [the proposed addition]
Tier 3: SSD/mmap  — full weight tensor, demand-paged        (already works today)
```

Results on RTX PRO 2000 (8 GB VRAM), GPT-OSS-120B:

| Phase              | tok/s    | Cache hit rate |
| ------------------ | -------- | -------------- |
| Cold start (0–80)  | 1.9–2.5  | 48–56%         |
| Warming (80–160)   | 2.6–5.1  | 53–78%         |
| Steady state (160+)| 12–14    | 98–100%        |

Once the cache is warm, decode is GPU-compute bound — the PCIe pipe is idle for most tokens.
Pure CPU-offload with no cache would run at roughly 0.5–1 tok/s on the same hardware.

## What llama.cpp already has

The infrastructure is 80% there:

1. **`--n-cpu-moe` in llama-bench** (`tools/llama-bench/llama-bench.cpp:332`) — already wires up MoE tensor routing to CPU. Not exposed in `llama-cli` or `llama-server`.

2. **Selective expert copy in `ggml_backend_sched_compute_splits()`** (`ggml/src/ggml-backend.cpp:1445–1564`) — already reads the expert ID tensor at runtime and copies only the used expert sub-rows from CPU to GPU. This is the exact point where a RAM-side cache lookup would go.

3. **`llama_mmap` with `posix_madvise`** — exists in `src/llama.cpp`. `MADV_WILLNEED` for prefetch and `MADV_DONTNEED` for page eviction are already wrapped. Explicit SSD-tier management is a small addition on top.

## Proposed design

### 1. Expose `--n-cpu-moe` in main CLI (~30 LOC)

Port the existing `llama-bench` parameter to `llama-cli` and `llama-server`. Zero new behaviour — just wires up a parameter that already exists in the engine.

### 2. Pluggable expert cache policy interface

The key design ask is that the eviction policy is not hardcoded. MoE routing distributions vary significantly across models (uniform vs highly skewed), and the right policy depends on the workload (prefill-heavy vs decode-heavy). A simple interface:

```cpp
// ggml/src/ggml-moe-cache.h  (sketch — naming should follow ggml conventions)
struct ggml_moe_cache_policy {
    const char * name;
    // called on hit: expert_id was found in cache at slot
    void (* on_hit)(void * ctx, int32_t expert_id, int32_t slot);
    // called on miss: return slot to evict (or -1 if free slot available)
    int32_t (* select_evict)(void * ctx);
    // called after miss is resolved: expert loaded into slot
    void (* on_load)(void * ctx, int32_t expert_id, int32_t slot);
    // opaque policy state
    void * ctx;
};

// Built-in policies
struct ggml_moe_cache_policy ggml_moe_cache_policy_lru  (int capacity);
struct ggml_moe_cache_policy ggml_moe_cache_policy_slru (int capacity);  // two-tier
struct ggml_moe_cache_policy ggml_moe_cache_policy_lfu  (int capacity);
struct ggml_moe_cache_policy ggml_moe_cache_policy_fifo (int capacity);
```

Callers select a policy by name via `--moe-expert-cache-policy lru|slru|lfu|fifo`.

### 3. Why SLRU over plain LRU (the recommended default)

MoE has a specific access pattern problem: long prefill sequences activate all experts once (uniform distribution), then decode settles into a small hot set. A plain LRU cache gets wiped by every prefill burst, forcing a cold-start penalty on every new request.

SLRU (Segmented LRU) uses two internal tiers — *probationary* (new admits, ~20% of capacity) and *protected* (re-accessed experts, ~80%). Eviction always hits the probationary tier first. An expert accessed twice moves to protected and is shielded from one-off bursts.

Additionally, a **frequency-gated admission** policy (admit to RAM cache only on second miss, not first) blocks cold prefill experts from entering the cache at all. Together these reduce cold-start pollution significantly.

In the Python PoC, SLRU vs LRU improved steady-state hit rate by ~8–15 percentage points on mixed prefill+decode workloads.

### 4. Explicit SSD-tier control (~60 LOC)

Currently the OS demand-pages expert weights from the mmap'd file on access. Making this explicit:
- `posix_madvise(ptr, len, POSIX_MADV_WILLNEED)` after each decode step for the same expert IDs (they're likely needed again next token)
- `posix_madvise(ptr, len, POSIX_MADV_DONTNEED)` when an expert is evicted from the RAM cache (reclaim OS page cache on memory-constrained machines)

This is a fire-and-forget hint — no correctness dependency.

## What I am asking for

I am primarily a Python developer. The PoC is in Python on HuggingFace — I can show this works and explain the design, but I am not confident writing production-quality C++ for llama.cpp internals, particularly around the ggml graph execution and backend scheduler.

**Looking for:**
- A C++ contributor familiar with `ggml_backend_sched_compute_splits()` to implement the RAM cache between the mmap layer and the GPU copy
- Feedback on whether the pluggable policy interface fits ggml conventions
- Any existing work in this direction I may have missed

Happy to provide benchmark data, answer design questions, and test implementations.

## Files a C++ implementer would need to study

| File | Why |
| ---- | --- |
| `ggml/src/ggml-backend.cpp:1445–1564` | Expert copy loop — cache lookup goes here |
| `tools/llama-bench/llama-bench.cpp:332,1144–1184` | Existing `--n-cpu-moe` parameter wiring |
| `src/llama.cpp:481–509` | MoE tensor CPU placement logic |
| `src/llama.cpp` (search `llama_mmap`) | madvise wrappers |

---

*Note: I used AI tools to help structure this RFC and to analyse the codebase. The benchmark data and the Python PoC are my own work. I reviewed and edited this text myself.*
