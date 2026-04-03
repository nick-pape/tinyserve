# Throughput Optimization Roadmap

Target: maximize single-token decode throughput on RTX PRO 2000 (8GB VRAM) + 64GB RAM + PCIe 5.0.

Current: 14 tok/s warm, 2 tok/s cold. Bottleneck: mmap page faults (20ms/miss, 92% of token time on misses).

## Priority 1: Pinned CPU Expert Pool

**Expected speedup:** 10x on miss path (20ms -> 1.5-2ms per miss). Cold start: 2 -> 8-12 tok/s. Warm path unchanged (already cache hits).

**Complexity:** ~80 LOC. Pure Python. No new dependencies.

**Correctness risk:** None. Same data, different memory backing. Verify: byte-identical output before/after on 100-token generation.

**Note on dead-end record:** Project memory records "Pinned CPU experts: needs 57GB, infeasible" -- that calculation assumed BF16 pinning. We pin in STORAGE format (MXFP4/FP8), not BF16. MXFP4 storage is ~4-6MB/expert. 1536 experts x 6MB = ~9GB pinned. Leaves 55GB for OS. Even if experts are 12MB stored, 18GB pinned leaves 46GB. Both feasible.

**Implementation in `generic_store.py`:**

1. After mmap population in `build()` and `from_safetensors()`, allocate a pinned tensor:
```python
pinned = torch.empty(num_layers, num_experts, store_layout.total_bytes,
                     dtype=torch.uint8).pin_memory()
pinned.copy_(data)  # one-time bulk copy, faults mmap pages once
```

2. Replace `self._data = data` (mmap-backed) with `self._data = pinned`. Drop the mmap reference.

3. `copy_to_buffer()` non-FP8 path now does `buf.packed.copy_(self._data[layer_idx, expert_idx], non_blocking=True)` from pinned memory -- pure DMA, no page faults, non_blocking actually works (mmap non_blocking is a lie because page faults block the CPU thread).

4. FP8 path: the per-buffer `_cpu_bf16_stages` are already pinned. Source data is now also pinned. CPU dequant (FP8->BF16) runs on pinned source, writes to pinned staging, then DMA to GPU. All fast.

5. Delete `self._mmap` reference and the tempfile. Expert data lives entirely in pinned RAM.

6. `prefetch()` method becomes a no-op permanently (madvise is irrelevant once data is pinned).

**Warmup cost:** One-time ~9-18GB memcpy at startup. At RAM bandwidth (~40 GB/s), takes <0.5s. Acceptable.

**File changes:**
- `generic_store.py`: `build()`, `from_safetensors()`, `__init__()` (add pinned flag), `copy_to_buffer()` (simplify non-blocking path)


## Priority 2: Layer-Ahead Expert Prefetch

**Expected speedup:** 1.5-2x on miss path by hiding H2D latency behind attention compute. Compound with P1: miss cost drops from 1.5ms (pinned DMA) toward 0ms (overlapped).

**Complexity:** ~120 LOC. Pure Python. Requires access to router weights one layer ahead.

**Correctness risk:** None if prefetch is purely speculative (loaded but not used if wrong). Verify: identical output with and without prefetch.

**Implementation:**

1. **Router pre-computation in `offloaded_model.py`:** In `offloaded_forward()`, after computing `top_idx` for layer N, the hidden state entering layer N's attention is available. Run layer N+1's router on it to predict next-layer expert IDs. This is approximate (attention hasn't run yet) but 70-80% accurate for prefetch.

2. **New method `GenericExpertPipeline.prefetch_experts(layer_idx, expert_ids)`:** Issues `copy_to_buffer()` calls on a dedicated prefetch stream for predicted experts. Does NOT touch the LRU cache -- just warms the double buffers.

3. **In `_execute_token_experts()`:** Before the miss path, check if the needed expert is already in a prefetch buffer. If yes, skip the H2D transfer.

4. **Simpler alternative (implement first):** In `_pipeline_experts()`, when processing expert i of layer N, simultaneously issue H2D for expert 0 of layer N+1 if the router prediction is available. This requires passing next-layer expert IDs into the pipeline.

**File changes:**
- `offloaded_model.py`: `offloaded_forward()` (run next-layer router speculatively)
- `generic_pipeline.py`: `_pipeline_experts()` (accept and use prefetch hints)


## Priority 3: 3-Stream Concurrent H2D

**Expected speedup:** Up to 2-3x on the per-expert H2D transfer (1.5ms -> 0.5-0.7ms with 3 concurrent streams splitting gate_proj/up_proj/down_proj). Only benefits miss path.

**Complexity:** ~60 LOC. Pure Python. Uses `torch.cuda.Stream`.

**Correctness risk:** Low. Streams are independent, synchronization via events. Verify: byte-identical buffer contents after 3-stream vs 1-stream copy.

**Implementation in `generic_store.py`:**

1. Add `_h2d_streams: list[torch.cuda.Stream]` (3 streams) created lazily on first `copy_to_buffer()` call.

2. In `copy_to_buffer()`, split the expert's flat buffer into 3 segments by tensor name (gate_up_proj, gate_up_proj_scales, down_proj, down_proj_scales). Launch each segment's `.copy_()` on a separate stream.

3. Record events on all 3 streams. Return a composite event or have the caller wait on all 3.

4. The pipeline's `transfer_stream` becomes a coordinator that waits on the 3 sub-stream events.

**Caveat:** PCIe is a serial bus. 3 streams don't give 3x bandwidth. But they DO help because:
- Each `copy_()` has CPU-side overhead (kernel launch, descriptor setup). Overlapping this overhead with actual DMA saves time.
- For the FP8 path, CPU dequant of segment 2 overlaps with DMA of segment 1.

Realistic gain: 1.3-1.5x, not 3x.

**File changes:**
- `generic_store.py`: `copy_to_buffer()` (split into 3-stream pattern)
- `generic_pipeline.py`: `_pipeline_experts()` (wait on composite event)


## Priority 4: Frequency-Aware Cache Policy (LHU)

**Expected speedup:** 2-4% fewer evictions than LRU. Small but free after implementation.

**Complexity:** ~60 LOC in `cache_policy.py`. Pure Python.

**Correctness risk:** None. Cache policy doesn't affect computation, only eviction order.

**Implementation in `cache_policy.py`:**

1. New `LHUPolicy(CachePolicy)` class combining:
   - LRU score: normalized recency (already tracked)
   - LFU score: activation count per (layer, expert) key
   - Precision weight: boost score for experts that are typically high-gating-weight (need metadata from router)

2. Scoring: `score = w1 * lru_rank + w2 * log(freq)`. Evict lowest score.

3. Register in `make_policy()` as `"lhu"`.

4. Weights `w1=0.7, w2=0.3` as defaults (tunable).

**File changes:**
- `cache_policy.py`: new `LHUPolicy` class, update `make_policy()`


## Priority 5: Expert Substitution (BuddyMoE)

**Expected speedup:** ~10% throughput gain at low cache rates (eliminates some miss penalties entirely).

**Complexity:** ~100 LOC + offline profiling script. Pure Python.

**Correctness risk:** Medium. Substituting one expert for another changes model output. Quality degradation bounded by buddy similarity, but must be validated with perplexity benchmarks.

**Implementation:**

1. **Offline profiling script** (`scripts/find_buddies.py`): Run model on calibration set. For each expert pair (same layer), compute cosine similarity of their output activations. Store pairs with similarity > 0.95.

2. **Buddy map** stored as `dict[tuple[int,int], tuple[int,int]]` mapping (layer, expert) -> (layer, buddy_expert).

3. **In `_execute_token_experts()`:** On cache miss, check if buddy is in VRAM cache. If yes, use buddy's weights instead of doing H2D transfer. Flag the output as approximate (for debugging).

4. **In cache policy:** Still allocate a slot and load the real expert. The buddy substitution is a latency optimization, not a cache optimization.

**File changes:**
- `generic_pipeline.py`: `_execute_token_experts()` (buddy check before miss path)
- New `scripts/find_buddies.py`


## Do NOT Implement

- **CPU expert compute:** 320ms/expert in Python, confirmed dead end. Needs C++/AVX2 which is out of scope.
- **Expert deferral:** Confirmed dead end -- produces garbage text due to routing approximation cascading across layers.
- **madvise WILLNEED:** Irrelevant after P1 (pinned pool eliminates mmap entirely). Skip it.
- **MoE-Gen module batching:** Requires batch>1, we are batch=1 decode.
- **INT4 low-precision miss path:** The 1.19-1.57x gain is modest and requires storing a second copy of all experts. Pinned pool (P1) already reduces miss cost by 10x, making the INT4 path's marginal benefit negligible.
- **torch.compile on experts:** Tested, 1.01x, not worth compile time.


## Execution Order and Expected Cumulative Performance

| Step | Optimization | Miss Cost | Estimated tok/s (cold) | Estimated tok/s (warm) |
|------|-------------|-----------|----------------------|----------------------|
| Baseline | mmap, LRU | 20ms | 2 | 14 |
| After P1 | Pinned pool | 1.5-2ms | 8-12 | 14 |
| After P2 | + Layer prefetch | 0-1ms (overlap) | 15-25 | 14 |
| After P3 | + 3-stream H2D | 0-0.7ms | 18-30 | 14 |
| After P4 | + LHU policy | same, fewer misses | 20-32 | 14-15 |

Warm-path throughput is GPU-compute bound at ~14 tok/s and won't change until we optimize the expert forward pass itself (Triton kernels, which are a separate workstream).

The realistic end state after P1-P3: **20-30 tok/s cold start, 14 tok/s warm** (warm unchanged because cache hits already bypass the miss path). The cold-to-warm transition window shrinks from ~80 tokens to ~20-30 tokens.

## Implementation Timeline

- **P1 (Pinned Pool):** 2-3 hours. Do this first. Measure before/after.
- **P2 (Layer Prefetch):** 4-6 hours. Requires careful integration with the offloaded forward loop.
- **P3 (3-Stream H2D):** 2-3 hours. Isolated change in store/pipeline.
- **P4 (LHU Policy):** 1-2 hours. Drop-in via existing policy abstraction.
- **P5 (BuddyMoE):** 4-6 hours including profiling script. Defer until P1-P3 are validated.
