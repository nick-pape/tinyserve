# Reusable GGUF Quantized Compute Kernels — Availability Survey

*2026-04-04. Searched: PyPI, GitHub, HuggingFace, arxiv.*

---

## TL;DR

**No fused dequant+matmul Triton kernel for GGUF K-quants exists.** There are dequant-only Triton kernels (ComfyUI-GGUF) and fused CUDA kernels (vLLM, llama.cpp). The closest reusable pieces are:

| What | Where | License | Fused? | Triton? | K-quant? |
|------|-------|---------|--------|---------|----------|
| llama.cpp MMQ/MMVQ | ggml-org/llama.cpp | MIT | Yes | No (CUDA) | All types |
| vLLM gguf_kernel.cu | vllm-project/vllm | Apache-2.0 | Yes (matvec) | No (CUDA) | All K-quants |
| ComfyUI-GGUF Triton dequant | city96/ComfyUI-GGUF PR #336 | Apache-2.0 | No (dequant only) | Yes | All K-quants |
| GPTQ-Triton fused matmul | fpgaminer/GPTQ-triton | Apache-2.0 | Yes | Yes | GPTQ only |
| torch int4pack (tinygemm) | PyTorch built-in | BSD-3 | Yes | No (CUDA) | INT4 pack only |

---

## Detailed Findings

### 1. ComfyUI-GGUF Triton Dequant (Best Starting Point)

**URL:** https://github.com/city96/ComfyUI-GGUF/pull/336
**License:** Apache-2.0
**Status:** Draft PR, not merged, experimental

Triton `@triton.jit` kernels that decompress Q2_K through Q6_K and Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 to FP16/BF16. **Dequant only — no matmul fusion.** ~1.2x real-world speedup over PyTorch dequant.

**Why it matters:** Contains the complete Triton bit-unpacking logic for all K-quant types. This is the hard part. To make a fused kernel, you inline this dequant into a matmul tile loop.

### 2. vLLM GGUF CUDA Kernels

**URL:** vllm-project/vllm `csrc/quantization/gguf/gguf_kernel.cu`
**License:** Apache-2.0
**Status:** Production, merged. "Highly experimental and under-optimized."

Fused dequant+matvec in CUDA for Q4_K, Q5_K, Q6_K, Q8_0, and I-quants. Handles the batch=1 decode case (matvec, not full GEMM). This is exactly our use case but it's CUDA C++, not Triton.

### 3. GPTQ-Triton (Fusion Template)

**URL:** https://github.com/fpgaminer/GPTQ-triton
**License:** Apache-2.0

Fused dequant+matmul in Triton for GPTQ (uniform 4-bit with group scales/zeros). Based on Triton matmul tutorial with dequant inlined into the tile loop. GPTQ's structure is simpler than K-quants but the **fusion architecture is the same pattern** we'd use.

### 4. llama.cpp MMQ/MMVQ

**License:** MIT
**Status:** Production, battle-tested, all quant types

Custom CUDA kernels — not usable from Triton. Would need PyTorch C++ extension wrapper. The MMVQ kernels handle exactly our case (batch=1 matvec on quantized weights).

### 5. torch._weight_int4pack_mm (tinygemm)

**Built into PyTorch.** Works on our GPU (Blackwell SM 12.0). But requires converting Q4_K → INT4 packed format first. Loses ~1 bit on asymmetric distributions.

### 6. Non-Starters

| Project | Why not |
|---------|---------|
| torchao | No GGML quant types |
| HF `kernels` package | MXFP4 only |
| HF transformers GGUF | Dequant at load time only, no compute |
| marlin | GPTQ/AWQ only, CUDA not Triton |
| GemLite | No GGML block quant support |
| mlc-llm / TVM | Different ecosystem entirely |
| ctransformers | Abandoned |

---

## Three Options for tinyserve

### Option A: llama.cpp CUDA kernels via PyTorch extension
- **Extract:** MMVQ kernels from `ggml-cuda/mmvq.cu` + type-specific dequant from `ggml-cuda/dequantize.cuh`
- **Wrap:** PyTorch C++ extension (`torch.utils.cpp_extension`)
- **Stay linked:** git submodule of ggml repo, build from source
- **Pros:** Battle-tested, all quant types, zero development of kernels
- **Cons:** C++ build complexity, CUDA-only (no AMD), harder to debug/tune
- **Effort:** Medium (wrapping existing code)

### Option B: Fused Triton kernel (novel work)
- **Base:** ComfyUI-GGUF Triton dequant (bit-unpacking) + GPTQ-Triton (fusion template)
- **Build:** Inline K-quant dequant into Triton matmul tile loop
- **Pros:** Pure Python/Triton, easy to maintain, portable
- **Cons:** Novel work, needs tuning per GPU, nobody has validated this combination
- **Effort:** High (writing + tuning a new kernel)

### Option C: torch int4pack with Q4K→INT4 conversion at cache fill
- **Use:** Existing `torch._weight_int4pack_mm` (tinygemm)
- **Convert:** Q4_K → float32 → INT4 pack once per cache miss (~1ms)
- **Pros:** Zero kernel development, reuses PyTorch built-in, works today
- **Cons:** Lossy conversion, 1ms overhead per miss, INT4 pack is 4x not 4.5x compression
- **Effort:** Low (3 files changed, reuse existing code)

---

## Recommendation

**Start with C (works today), extract A for production quality.**

C unblocks all GGUF models immediately. A gives native-format compute with zero conversion loss. B is the most elegant but requires writing a novel kernel nobody has published.

For staying linked to llama.cpp: `git submodule add` the ggml repo, write a thin PyTorch extension that calls MMVQ kernels. Update submodule periodically to get upstream kernel improvements.

---

## Sources

1. [ComfyUI-GGUF Triton PR #336](https://github.com/city96/ComfyUI-GGUF/pull/336)
2. [vLLM GGUF CUDA Kernel](https://github.com/vllm-project/vllm/tree/main/csrc/quantization/gguf)
3. [GPTQ-Triton](https://github.com/fpgaminer/GPTQ-triton)
4. [llama.cpp MMVQ](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu)
5. [PyTorch AO HQQ Fused PR](https://github.com/pytorch/ao/pull/153)
6. [W4A16 SplitK (arxiv 2402.00025)](https://arxiv.org/abs/2402.00025)
7. [GemLite](https://github.com/mobiusml/gemlite)
8. [transformers-qwen3-moe-fused](https://github.com/woct0rdho/transformers-qwen3-moe-fused)
