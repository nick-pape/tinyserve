# tinyserve Architecture

## Overview

tinyserve serves Mixture-of-Experts (MoE) language models on consumer GPUs by keeping
most expert weights in CPU RAM (or on disk) and streaming only the experts needed for
each forward pass into a fixed VRAM cache. The design targets single-GPU inference where
VRAM fits neither the full model nor a na&iuml;ve per-expert copy.

## Data Flow

### Model Loading

1. The caller invokes `load_and_offload(model_path, config)` or `offload_model(model, config)`.
2. tinyserve queries free VRAM, applies the `gpu_memory_utilization` cap, and computes
   how many expert weight blocks fit in the remaining budget.
3. Expert tensors are extracted from the HuggingFace model and packed into a flat
   `ExpertStore` backed by a memory-mapped file or RAM buffer.
4. Non-expert modules (embedding, attention, layer-norm, LM head) stay on GPU in BF16
   or FP8 as usual.
5. An `ExpertPipeline` is created per transformer layer, sharing a pair of double-buffered
   staging tensors for async DMA transfers.
6. `OffloadedLM` wraps the patched model together with the pipelines, `StaticKVCache`,
   and `VRAMBudget` into a single typed object.

### Forward Pass

Each transformer layer's MoE block runs through `ExpertPipeline.execute_layer_experts`:

```
Token hidden states
       |
       v
  MoE Router  -->  top-K expert IDs + weights  (unchanged HF code)
       |
       v
  ExpertCache.lookup(layer, expert_id)
       |
    HIT? ──── yes ──> expert weight block already in VRAM
       |                        |
      no                        |
       |                        v
  DMA transfer from       GPU MatMul (gate_up_proj + SiLU + down_proj)
  ExpertStore via                |
  async CUDA stream              |
       |                        |
       +------------------------+
       |
       v
  Weighted sum of expert outputs
       |
       v
  Layer output
```

If `cpu_on_miss=True`, cache misses fall back to `CPUExpertForward` — the CPU computes
the expert output and sends the result to GPU, skipping the DMA load for that token.
This is faster when the DMA queue is saturated.

Prefill (multi-token) uses `execute_layer_experts_batched`, which groups tokens by
expert assignment, loads each expert once, and dispatches all tokens for that expert
as a single batched matmul.

### Cache Management (LFRU)

`ExpertCache` owns a flat VRAM tensor divided into fixed-size slots (one per expert).
It delegates eviction decisions to a pluggable `CachePolicy`. The default policy is
`LFRUPolicy` (Least-Frequency-Recency-Utilized):

- Each cached entry carries a `freq` count (incremented on every hit) and a `clock`
  timestamp (updated on every access).
- Eviction score = `freq / age` — entries that are accessed often and recently score
  highest and are kept.
- This dominates pure LRU (which would evict hot-but-recently-displaced experts) and
  pure LFU (which would keep stale prefill experts that are never accessed again in
  decode).
- A Cython hot path (`_fast_cache.pyx`) accelerates the `select_evict` scan for large
  caches.

Optional: `BuddyTable` pre-fetches the statistically most co-activated expert alongside
every cache miss, using a co-activation matrix built from routing history. This halves
cold-start misses on repetitive text patterns.

### VRAM Budget

`VRAMBudget` runs a lightweight check after each token:

- If `StaticKVCache` utilization exceeds a high-water mark, it calls `cache.shrink(n)`
  to evict experts and donate slots to KV memory.
- If KV utilization drops below a low-water mark after a request completes, it calls
  `cache.grow(n)` to reclaim slots for experts.

This keeps total VRAM usage below `gpu_memory_utilization × total_VRAM` without manual
tuning.

## Key Classes

| Class | File | Responsibility |
|---|---|---|
| `TinyserveConfig` | `offload.py` | Dataclass holding all runtime options (cache size, policy, VRAM cap, dtype, etc.) |
| `OffloadedLM` | `offload.py` | Public wrapper returned by `load_and_offload`; holds model + pipelines + KV cache + budget |
| `OffloadedModel` | `_model_hooks.py` | Patches MoE forward hooks into a HuggingFace model in-place |
| `ExpertStore` | `expert_store.py` | Flat mmap-backed buffer of all expert weight blocks; provides async DMA to GPU staging buffers |
| `ExpertCache` | `expert_cache.py` | VRAM slot allocator; tracks hits/misses per layer; delegates eviction to a `CachePolicy` |
| `ExpertPipeline` | `expert_pipeline.py` | Per-layer orchestrator: cache lookup → DMA transfer → GPU compute → weighted sum |
| `LFRUPolicy` | `cache_policy.py` | Default eviction policy: frequency/recency score, Cython-accelerated hot path |
| `BuddyTable` | `buddy_experts.py` | Co-activation matrix + buddy prefetch; reduces cold-start misses |
| `StaticKVCache` | `static_kv_cache.py` | Pre-allocated KV cache with static shapes for `torch.compile`-friendly inference |
| `VRAMBudget` | `vram_budget.py` | Monitors expert + KV VRAM pressure; triggers `shrink`/`grow` on `ExpertCache` |
| `CPUExpertForward` | `cpu_expert.py` | CPU fallback for cache misses when PCIe DMA is saturated |
