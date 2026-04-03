# Throughput Optimization Roadmap v2

Updated 2026-03-20. Incorporates findings from 3-agent research sweep across vLLM/llama.cpp PRs,
research papers (SpecMD, HOBBIT, FATE, DALI, ExpertFlow, MoE-SpeQ, Fiddler) and CUDA perf analysis.

## Hardware Reality Check (corrected)

- RTX PRO 2000: **PCIe 5.0 x8**, not x16 → **~31.5 GB/s unidirectional H2D** (not 64 GB/s)
- Expert size: **12.65 MB** per expert (from EXPERT_BYTES)
- Theoretical DMA floor: 12.65 MB / 31.5 GB/s = **0.4 ms per expert**
- With top_k=4 and double buffering: 2 serial DMAs × 0.4 ms = **0.8 ms per MoE layer** minimum cold
- GPU internal bandwidth: 288 GB/s (GDDR7) — VRAM→VRAM copies: 12.65 MB / 288 GB/s = **44 µs**
- Theoretical cold ceiling: ~34 tok/s (PCIe-bound)
- Warm path is **memory-bandwidth bound** (2 FLOPs/byte vs ridge point of ~1892 FLOPs/byte)

## Completed

| Step | Optimization | Status |
|------|-------------|--------|
| P1 | Pinned CPU expert pool | **Done** (commit 0c6a3fb) |
| P2 | Temporal layer prefetch (~75% accuracy) | **Done** |
| P3 | 3-stream concurrent H2D | **Done** |
| P4 | LFRU cache policy | **Done** |

## Immediate Priority (< 100 LOC each, high confidence)

### I1: Verify Pinned Memory (10 LOC, critical diagnostic)

**Why:** If `self._data.is_pinned()` returns False, P1 never worked. CUDA routes unpinned memory
through an internal staging copy, doubling effective bandwidth needed. This single bug would
explain cold performance below expectations.

**Implementation:** Add assertion in `GenericExpertStore.__init__`:
```python
assert self._data.is_pinned(), "Expert store must be pinned for async H2D"
```
Also log `store._data.is_pinned()` at startup in `offload.py`.

### I2: Param Pointer Swap instead of VRAM Copy (5 LOC)

In `generic_pipeline.py` → `swap_weights_and_forward()`, replace:
```python
getattr(mod, parts[-1]).copy_(buf.get_tensor(name))  # VRAM→VRAM copy, 44 µs/param
```
with:
```python
getattr(mod, parts[-1]).data = buf.get_tensor(name)  # pointer remap, ~0 µs
```
Works because buf tensors remain valid for the forward pass duration. Safe inside `torch.no_grad()`.
**Expected gain:** Eliminates N × 44 µs per expert for N params (2–4 params → 0.1–0.2 ms/expert).

### I3: Least-Stale Eviction Policy (~200 LOC, drop-in)

**Paper:** SpecMD (arxiv 2602.03921). Key insight: MoE access within one token is sequential and
deterministic. After layer N fires an expert, it won't be re-accessed this token. LRU/LFRU treat
it as "recently used" and keep it — wrong. Least-Stale evicts stale-first.

**Result:** 1.6–1.9% collision rate at 5% cache capacity vs LRU's 4.5–12.6%. 85× fewer collisions.

**Implementation in `cache_policy.py`:**
- Track `_current_pass_keys: set` of experts accessed THIS forward pass
- On `lookup`: mark key as accessed in current pass
- On `select_evict`: prefer stale keys (not in current pass), tie-break by FIFO layer order
- On new token: call `policy.begin_pass()` to rotate current→stale

### I4: Cache-Aware Routing / Logit Bias (~30 LOC)

**Papers:** ExpertFlow (arxiv 2510.26730), SpecMD. Add small bias to router logits for GPU-resident
experts before top-k. Steers routing toward cache hits without weight changes.
ExpertFlow reports 96.65% latency reduction on DeepSeek/Qwen.

**Implementation in `offloaded_model.py` → `offloaded_forward()`:**
After router produces logits (before top-k), add `CACHE_BIAS` to logits of cached experts.
Bias magnitude: `mean(|logits|) × α` where α ∈ [0.1, 0.3]. Tunable. Perplexity validation required.

**Correctness risk:** Changes routing. Run perplexity benchmark before/after to confirm acceptable.

## Short-Term Priority (100–400 LOC)

### S1: FATE Cross-Layer Gate Prefetch (~300 LOC) [P2 upgrade]

**Paper:** arxiv 2502.12224. Run the **actual** next-layer gate on CPU concurrently with GPU
attention. Achieves **97.15% prefetch accuracy** vs our ~75%. 4.5× decode speedup on RTX 3090.

Gate weights already pinned on CPU from P1. CPU gate GEMV is tiny (hidden×num_experts floats).

**Implementation:** Background CPU thread receives current hidden state after routing, applies
next-layer gate weights, returns expert predictions. Results fed into `schedule_prefetch()`.
Threading model: Python `threading.Thread` with queue handoff.

**Key advantage over P2:** Uses ACTUAL next-layer gate, not temporal approximation. 97% vs 75%
prefetch accuracy → 25%+ more cache hits during cold-start phase.

### S2: Triton BLOCK_N Autotuner (~20 LOC) [warm path]

In `triton_dot_scaled.py`, add `@triton.autotune` with `BLOCK_N ∈ {32, 64, 128}`.
On Blackwell sm_120 with GDDR7, larger tiles may better hide memory latency.
**Expected gain: 10–30% warm path → 14 tok/s → 15.4–18.2 tok/s**.

### S3: NSys Profiling Pass

Before coding further, profile with:
```bash
nsys profile --trace cuda,nvtx python benchmark.py
```
Key metrics: H2D transfer bandwidth (are we hitting 31.5 GB/s?), kernel gaps (CPU overhead),
L2 cache miss rate (weight layout issue?). This tells us where the remaining headroom is.

## Medium Priority (research-validated, more complex)

### M1: DALI Workload-Aware Sliding Window Cache (~200 LOC)

**Paper:** arxiv 2602.03495. Sliding window of per-expert token counts (last K=256 tokens).
High-workload experts (many tokens routed to them) stay cached regardless of recency.
Achieves **100% hit rate at steady state** on many workloads. 3.97× vs llama.cpp.

Can combine with Least-Stale: Least-Stale handles inter-token eviction, sliding window provides
the frequency signal.

### M2: INT4 Miss-Path (HOBBIT, ~400 LOC)

**Paper:** arxiv 2411.01433. On cache miss for low-importance experts, load INT4 version
(4× smaller → 4× faster H2D). Gating weight magnitude predicts importance (0.99 Pearson).
High-importance experts load FP16; low-importance load INT4.

**On PCIe 5.0 x8:** INT4 expert = 3.16 MB → 0.1 ms DMA. Significant miss-path speedup.
Requires offline INT4 quantization + second CPU memory store (+25% RAM: 13 GB vs 10 GB).

### M3: Multi-Token Speculative Prefetch (MoE-SpeQ style, ~400 LOC)

Current P2 prefetches experts for the NEXT token using current routing as prediction.
Multi-token lookahead: issue H2D for k=4–6 tokens ahead simultaneously.
INT4 quantized routing prediction: 90.9% accuracy (MoE-SpeQ). Hides k×DMA latency.

## Do NOT Implement

- **CPU expert compute (Python):** 320ms/expert. AMX/AVX512 C++ is needed; Fiddler's
  advantage shrinks with MXFP4 compression + PCIe 5.0 (0.4ms DMA vs 0.27ms CPU = too close).
- **Expert Deferral (KTransformers):** Requires C++/AMX kernel infrastructure.
- **FloE INT2+sparsity:** 4.4–7.6% accuracy degradation — unacceptable.
- **Expert result caching:** KV cache already handles this; continuous hidden states don't repeat.
- **CUDA stream priority:** No effect on DMA copy engines (hardware-documented limitation).
- **Direct cuMemcpyAsync:** <1% gain vs PyTorch, not worth the fragility.
- **madvise WILLNEED:** Irrelevant since P1 (data fully pinned, no page faults).

## Revised Performance Projections

| State | Cold tok/s | Warm tok/s | Notes |
|-------|-----------|-----------|-------|
| Baseline (mmap, LRU) | 2 | 14 | |
| After P1–P4 (now) | 8–12 | 14 | Estimate, not yet benchmarked |
| + I2 (param swap) | 9–14 | 14.5 | Eliminates VRAM-VRAM copy overhead |
| + I3 (Least-Stale) | 10–16 | 14.5 | Fewer cache collisions |
| + I4 (logit bias) | 12–20 | 14.5 | Steers routing toward hits |
| + S1 (FATE 97%) | 16–25 | 14.5 | Near-zero miss after token 2 |
| + S2 (Triton tune) | 16–25 | 17–18 | Warm path improvement |
| PCIe-bound ceiling | ~34 | ~34 | 4 experts × 0.4ms DMA, double-buf |

**Warm path ceiling note:** At 14 tok/s (71 ms/tok), increasing warm path requires either:
1. Triton kernel tuning (BLOCK_N autotune, estimated +10–30%)
2. Fused gate+silu+down kernel (saves 1 VRAM round-trip of intermediate, marginal)
3. Weight tensor layout optimization (column-major within tile, profile first)

## Execution Order

1. **Benchmark on GPT-OSS-20B first** (need actual baseline numbers before optimizing further)
2. **I1: Verify pinning** (10 min, critical sanity check)
3. **I2: Param swap** (30 min, 5 LOC, no correctness risk)
4. **I3: Least-Stale policy** (2–3 hrs, drop-in, no correctness risk)
5. **I4: Logit bias** (1 hr + perplexity validation)
6. **S3: NSys profile** (after each batch to understand remaining headroom)
7. **S1: FATE prefetch** (4–6 hrs, major P2 upgrade)
8. **S2: Triton autotune** (1 hr)
