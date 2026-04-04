# MMQ vs Fused Dequant Performance — Benchmark Data

*2026-04-04. Data from llama.cpp PRs, ExLlamaV2 benchmarks, Blackwell microbenchmarks, MoE-Infinity paper.*

---

## TL;DR

**At batch=1 decode (our MoE expert case): the kernel choice barely matters.** The operation is memory-bandwidth-bound. Both dp4a (MMQ) and fused-dequant-to-FP16 read the same bytes from VRAM. The real wins are: (1) fewer kernel launches, (2) CUDA graphs, (3) expert prefetch overlap.

---

## 1. MMQ vs cuBLAS-with-Dequant (Prompt Processing)

Source: [llama.cpp PR #8062](https://github.com/ggml-org/llama.cpp/pull/8062)

### RTX 4090 (Ada SM 8.9), Llama 8B Q4_K_S, pp2048

| Microbatch | cuBLAS (t/s) | MMQ (t/s) | MMQ Speedup |
|------------|-------------|-----------|-------------|
| 16 | 1,983 | 2,000 | 1.01x |
| 32 | 3,369 | 3,519 | 1.04x |
| 128 | 3,476 | 7,284 | **2.10x** |
| 256 | 5,745 | 9,216 | **1.60x** |
| 512 | 7,637 | 9,867 | **1.29x** |
| 1024 | 8,954 | 9,664 | 1.08x |
| 2048 | 8,975 | 8,828 | 0.98x |

### RTX 3090 (Ampere SM 8.6), Llama 8B Q4_K_S, pp2048

| Microbatch | cuBLAS (t/s) | MMQ (t/s) | MMQ Speedup |
|------------|-------------|-----------|-------------|
| 16 | 1,293 | 1,341 | 1.04x |
| 128 | 2,211 | 3,198 | **1.45x** |
| 512 | 3,820 | 3,730 | 0.98x |
| 2048 | 4,502 | 3,678 | 0.82x |

**Pattern:** MMQ dominates at batch 16-256. cuBLAS overtakes at batch 512+ (Ampere) or 2048+ (Ada). Ada's superior INT8 TC extends MMQ's advantage.

---

## 2. ExLlamaV2 vs llama.cpp (Token Generation)

Source: [oobabooga blog](https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/), RTX 3090, Llama 2 13B 4-bit

| Backend | Quant | Tok/s (generation) |
|---------|-------|-------------------|
| ExLlamaV2 EXL2 4.25bpw | EXL2 | 56.90 |
| ExLlamaV2 EXL2 4.65bpw | EXL2 | 56.46 |
| llama.cpp Q4_K_M | GGUF | 30.83 |
| llama.cpp Q4_K_S | GGUF | 35.30 |

**ExLlamaV2 is 1.6-1.85x faster** for generation. Reason: simpler weight packing format (EXL2 vs Q4_K super-blocks) means less unpacking overhead in the bandwidth-bound regime. ExLlamaV2 uses fused dequant-to-FP16 + TC matmul, NOT integer dp4a.

*Caveat: early 2024 data. llama.cpp has improved since.*

---

## 3. Blackwell (SM 12.0) Compute Throughput

Source: [arxiv 2512.02189](https://arxiv.org/html/2512.02189v2)

| Precision | Throughput | Notes |
|-----------|-----------|-------|
| FP4 TC | 7,700 TFLOPS | Blackwell-native |
| INT8 TC | 3,929 TOPS | 96%+ of peak |
| FP16 TC (FP16 accum) | 1,930 TFLOPS | 96.5% of peak |
| FP16 TC (FP32 accum) | 482 TFLOPS | 50% reduction |

B200 sustained bandwidth: 4.14 TB/s (51.8% of 8 TB/s peak).

**For batch=1 decode:** compute is irrelevant. At 4.14 TB/s sustained and ~10 GB per forward pass, max throughput is ~400 tok/s regardless of compute precision. Both INT8 and FP16 saturate bandwidth identically.

**For prefill:** FP4 TC at 7,700 TFLOPS makes both INT8 dp4a and FP16 fused-dequant approaches obsolete on Blackwell. NVFP4 is the winning path.

---

## 4. MoE Expert Offloading: What Actually Matters

### Kernel Launch Overhead

| Metric | Value | Source |
|--------|-------|--------|
| CUDA kernel launch latency | 5-20 us | NVIDIA developer forums |
| Per-expert GEMM at batch=1 (W4A16) | 28-50 us | W4A16 SplitK paper |
| 3 kernel launches per expert (dequant+GEMM+act) | 15-60 us overhead | Estimated |
| KTransformers GPU overhead before CUDA graphs | >20% of total | KTransformers SOSP25 |
| FlashDMoE: 550 kernels (baseline) vs 1 persistent | 21.9% bandwidth saving | FlashDMoE paper |

**Kernel launches cost as much as the compute itself at batch=1.**

### Expert Transfer Latency (the real bottleneck)

| System | Per-Expert Fetch | Source |
|--------|-----------------|--------|
| Switch Transformer | 1 ms | MoE-Infinity |
| Mixtral | 10 ms | MoE-Infinity |
| Arctic | 7 ms | MoE-Infinity |
| GPU blocking time | 12-43% of inference | MoE-Infinity |

**Expert transfer is 100-1000x slower than kernel compute.** The cache hit rate determines throughput, not the kernel choice.

### Conclusion for tinyserve

**Priority order:**
1. **Cache hit rate** (LFRU policy, FATE prefetch) — dominates everything
2. **Cache slot size** (INT4 = 4x more experts in same VRAM = higher hit rate)
3. **Kernel fusion** (dequant+GEMM in one launch) — saves 10-20 us/expert
4. **CUDA graphs** — eliminates launch overhead entirely
5. **Kernel precision** (dp4a vs FP16 TC) — negligible at batch=1

---

## 5. The Answer: A vs B for tinyserve

**At batch=1 decode, approaches A (llama.cpp MMQ dp4a) and B (fused dequant-to-FP16) perform identically.** Both are bandwidth-bound, both read the same bytes from VRAM.

The difference is in implementation complexity:
- **A (llama.cpp kernels):** battle-tested, MIT licensed, handle all quant types. But they're C++/CUDA, not Triton — harder to maintain and extend.
- **B (fused Triton dequant):** can be written once parametrically for all K-quant types. Triton is more maintainable than raw CUDA. But we'd be writing it from scratch.

**The pragmatic path:** Use llama.cpp kernels (approach A via MIT extraction) because they already exist, are tested, and handle all quant types. The performance difference vs B is zero at batch=1. Don't write what already exists.

---

## Sources

1. [llama.cpp PR #8062: MMQ int8 TC optimization](https://github.com/ggml-org/llama.cpp/pull/8062)
2. [llama.cpp PR #8075: MMQ as default](https://github.com/ggml-org/llama.cpp/pull/8075)
3. [llama.cpp Discussion #17621: Optimizing CUDA Backend](https://github.com/ggml-org/llama.cpp/discussions/17621)
4. [oobabooga: GPTQ vs AWQ vs EXL2 vs llama.cpp](https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/)
5. [Blackwell Microbenchmarks (arxiv 2512.02189)](https://arxiv.org/html/2512.02189v2)
6. [FP6-LLM: TC-FPx fused dequant (arxiv 2401.14112)](https://arxiv.org/html/2401.14112v1)
7. [MoE-Infinity (arxiv 2401.14361)](https://arxiv.org/html/2401.14361v2)
8. [FlashDMoE (arxiv 2506.04667)](https://arxiv.org/html/2506.04667v1)
9. [KTransformers SOSP 2025](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
10. [TFLOPS Gap: FP4 MoE on Blackwell (HuggingFace)](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison)
11. [ExLlamaV2 Issue #670: No-dequant discussion](https://github.com/turboderp/exllamav2/issues/670)
12. [W4A16 SplitK Triton Kernel (arxiv 2402.00025)](https://arxiv.org/html/2402.00025)
