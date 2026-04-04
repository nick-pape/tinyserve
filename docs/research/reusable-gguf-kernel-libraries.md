# Reusable GGUF Kernel Libraries — Complete Survey

*2026-04-04. Searched: PyPI, GitHub, HuggingFace, arxiv.*

---

## The Winner: sgl-kernel

**`pip install sgl-kernel`** — Apache 2.0, actively maintained (v0.3.21, 64+ releases), provides fused CUDA matmul for all GGUF K-quant types, callable directly with PyTorch tensors.

```python
from sgl_kernel.quantization.gguf import (
    ggml_dequantize,      # weight tensor -> dequantized FP16/BF16
    ggml_mul_mat_a8,      # fused quantized matmul (batched)
    ggml_mul_mat_vec_a8,  # fused quantized mat-vec (single token decode)
    ggml_moe_a8,          # MoE expert matmul (!)
)
```

**Supports:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ types.

**Architecture:** CUDA kernels (`gguf_kernel.cu`, `mmq.cuh`, `mmvq.cuh`, `vecdotq.cuh`) registered as `torch.ops.sgl_kernel.*`. Same kernel code as vLLM's GGUF support, but packaged as a lightweight standalone library.

**Caveat:** Requires `torch==2.9.1` pin. Need to verify compatibility with our environment.

---

## Full Ranking

| Rank | Library | Fused Matmul? | All K-quants? | Install | License |
|------|---------|--------------|---------------|---------|---------|
| **1** | **sgl-kernel** | Yes (CUDA) | Yes | `pip install sgl-kernel` | Apache-2.0 |
| **2** | **city96/ComfyUI-GGUF dequant.py** | No (dequant only) | Yes | Copy 300 LOC | Apache-2.0 |
| **3** | **vLLM gguf_kernel.cu** | Yes (CUDA) | Yes | Extract from vLLM | Apache-2.0 |
| **4** | **llama.cpp MMVQ** | Yes (CUDA) | Yes | Submodule + wrap | MIT |
| **5** | **torch int4pack** | Yes (built-in) | No (needs Q4K→INT4 conversion) | Built-in | BSD-3 |

---

## What Each Option Actually Is

### sgl-kernel (SGLang)
- Extracted kernel library from SGLang inference engine
- Same CUDA code as vLLM's GGUF support, lighter package
- Has `ggml_moe_a8` — **dedicated MoE expert matmul function**
- `torch.ops` registration means compatible with `torch.compile`
- Active: 64+ releases, last updated recently

### city96/ComfyUI-GGUF dequant.py
- Pure PyTorch dequant (~300 LOC), zero compilation
- Works on CPU and GPU, any platform
- **Dequant only** — you get FP16 weights, then do regular matmul
- Good for prototyping, bad for production (double bandwidth)
- Triton extension in draft PR #336 adds ~1.2x speedup (still dequant-only)

### vLLM gguf_kernel.cu
- Same kernels as sgl-kernel but embedded in vLLM's massive build
- Can't pip install separately without all of vLLM
- Could extract ~8 CUDA files and compile as standalone PyTorch extension
- "Highly experimental and under-optimized" per vLLM docs

### llama.cpp MMVQ
- Battle-tested, all quant types, MIT licensed
- Pure CUDA C++, not structured as PyTorch ops
- Would need custom PyTorch C++ extension wrapper
- git submodule to stay linked to upstream

### torch._weight_int4pack_mm (tinygemm)
- Built into PyTorch, zero dependencies
- Only handles INT4 packed format — Q4_K needs conversion first
- Conversion is lossy (~1 bit) and costs ~1ms per expert
- Already works in our codebase (GPUINT4Forward)

---

## Non-Starters

| Library | Why Not |
|---------|---------|
| torchao | No GGML quant types |
| HF `kernels` | MXFP4 only |
| marlin | GPTQ/AWQ only |
| GemLite | No GGML block quant |
| ggml-python | Dormant, no PyTorch integration |
| llama-cpp-python | High-level inference only |
| ctransformers | Abandoned |
| mlc-llm / TVM | Different ecosystem |

---

## Recommendation for tinyserve

**Use sgl-kernel as primary, city96 dequant.py as fallback.**

```
sgl-kernel available?
├── Yes → ggml_mul_mat_vec_a8() for decode, ggml_moe_a8() for MoE
│         Zero conversion, native K-quant format in GPU cache
│         4.5x smaller cache slots than BF16
└── No  → city96 dequant.py → FP16 → F.linear
          Works everywhere, slower (2x bandwidth)
```

This gives us:
- **Zero kernel development** — pip install
- **All GGUF quant types** — Q2_K through Q6_K, legacy, IQ types
- **Native format in GPU cache** — no conversion loss, minimum bytes
- **Dedicated MoE function** — `ggml_moe_a8` designed for our exact use case
- **Graceful fallback** — city96 dequant works without CUDA compilation

---

## Sources

1. [sgl-kernel on PyPI](https://pypi.org/project/sgl-kernel/)
2. [SGLang GGUF implementation](https://github.com/sgl-project/sglang)
3. [city96/ComfyUI-GGUF dequant.py](https://github.com/city96/ComfyUI-GGUF/blob/main/dequant.py)
4. [ComfyUI-GGUF Triton PR #336](https://github.com/city96/ComfyUI-GGUF/pull/336)
5. [vLLM GGUF kernels](https://github.com/vllm-project/vllm/tree/main/csrc/quantization/gguf)
6. [llama.cpp MMVQ](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu)
7. [GPTQ-Triton](https://github.com/fpgaminer/GPTQ-triton)
