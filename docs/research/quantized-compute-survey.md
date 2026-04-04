# Quantized Compute for MoE Expert Offloading — Research Survey

*2026-04-04. Sources: llama.cpp source, ExLlamaV2 author statements, PyTorch docs, KTransformers SOSP paper, PowerInfer paper.*

---

## 1. How llama.cpp Does It

Three GPU paths, selected automatically:

### MMQ (Matrix-Matrix Quantized) — Default on Ampere+
- **Weights stay in native Q4_K format in VRAM** — zero conversion
- Activations quantized on-the-fly to `block_q8_1_mmq` format
- Custom CUDA kernels perform **fused dequant + integer dot products** using `dp4a` (int8 dot-product-accumulate)
- Dequantization happens inline as data streams from shared memory tiles — no separate pass
- One kernel per quant type (`load_tiles_q4_K`, `vec_dot_q4_K_q8_1_dp4a`, etc.)

### cuBLAS (Dequant-then-GEMM) — Default on V100/RDNA3
- Bulk dequant Q4_K → FP16 into temporary buffer via `dequantize_row_q4_K`
- Then `cublasGemmEx` for FP16 tensor core matmul
- Uses more memory, can be faster for large batch prefill on datacenter GPUs
- Force with `GGML_CUDA_FORCE_CUBLAS`

### MMVQ (Matrix-Vector Quantized) — Batch <= 8
- Specialized for tiny batches (decode)
- Multiple warps reduce partial products for a single row
- Also fuses dequantization

**Selection logic:** MMQ when `__dp4a` or int8 TC available. cuBLAS when only FP16 TC exist. MMVQ for very small batches.

---

## 2. How ExLlamaV2 Does It

- Weights stored in EXL2/GPTQ format in VRAM
- **Fused dequant-to-FP16 inside the matmul kernel** — weights dequantized on-the-fly in registers/shared memory before multiply-accumulate
- NOT true integer compute — it's fused dequant + FP16 tensor core matmul
- EXL2 supports mixed 2/3/4/5/6/8-bit per layer with fractional average bitwidths
- **40-70% faster than llama.cpp** on equivalent NVIDIA hardware (per author benchmarks)

Key quote from turboderp: *"conversion to FP16 at some point... happens in the matmul kernel on individual weights as they're being streamed from VRAM."*

---

## 3. torch._weight_int4pack_mm

**Completely different format from Q4_K.** Requires conversion.

- uint4 asymmetric per-group quantization
- `TensorCoreTiledLayout` with `inner_k_tiles=8` — layout specifically for NVIDIA TC access patterns
- Backed by **tinygemm** kernel (batch=1 optimal) and **GemLite** (larger batches)
- Group sizes: 32, 64, 128, or 256
- **Best at batch=1** (memory-bandwidth-bound decode) — exactly our use case

To use Q4_K weights: dequant Q4_K → float32 → requant to int4pack. Loses ~1 bit effective precision on asymmetric distributions. Acceptable for 4-bit.

---

## 4. MoE Expert Storage in Production Systems

| System | Expert Storage Format | Where Experts Live | Compute |
|--------|----------------------|-------------------|---------|
| **llama.cpp** | Native GGUF quant (Q4_K etc.) | GPU: native format, CPU: native format | MMQ/MMVQ custom kernels |
| **PowerInfer** | Custom PowerInfer GGUF | Hot neurons on GPU (native quant), cold on CPU | Per-neuron activation, GGUF kernels |
| **KTransformers** | GGUF → AMX-optimized INT4/INT8 at load | Experts on CPU (AMX format), dense layers on GPU | AMX INT4/INT8 matmul on CPU |
| **ExLlamaV2** | EXL2/GPTQ in VRAM | All on GPU | Fused dequant+FP16 TC matmul |

**Key insight:** llama.cpp and PowerInfer keep native quant format everywhere. KTransformers converts once at load to a compute-optimized format. ExLlamaV2 keeps its own format.

---

## 5. Performance Data

### Generation (batch=1, memory-bandwidth-bound)
- Format that minimizes VRAM reads wins
- Native quant (MMQ, EXL2 fused) excels
- RTX 5090: Q4_K_M is **8% faster** than MXFP4 for generation

### Prefill (large batch, compute-bound)
- Tensor core throughput matters more than bandwidth
- Dequant-to-FP16 + cuBLAS can win
- RTX 5090: MXFP4 is **15% faster** than Q4_K_M for prefill

### torch int4pack (tinygemm)
- Best at batch=1 — designed for this regime
- Efficiency declines at larger batches where GemLite takes over

---

## 6. GGUF Quantization Types — What Matters

### Most Used (support these first)
| Type | bpw | Format | Notes |
|------|-----|--------|-------|
| **Q4_K_M** | 4.5 | 256-weight super-blocks, 6-bit scales | **Dominant.** Most downloaded on HuggingFace |
| **Q5_K_M** | 5.3 | 256-weight super-blocks, 6-bit scales | Quality sweet spot |
| **Q8_0** | 8.5 | 32-weight blocks, FP16 scale | Near-lossless reference |
| **Q6_K** | 6.6 | 256-weight super-blocks, 8-bit scales | Used internally by Q4_K_M for attn/output layers |

### Important (support next)
| Type | bpw | Notes |
|------|-----|-------|
| **Q4_K_S** | 4.4 | All-Q4_K variant (no mixed precision) |
| **Q5_K_S** | 5.2 | All-Q5_K variant |
| **IQ4_XS** | 4.25 | Better than Q4_K_M at similar size |
| **Q3_K_M** | 3.5 | Low-VRAM option |
| **Q2_K** | 3.2 | Extreme compression |

### K-quant Mixed Precision
**_S/_M/_L suffixes control per-tensor precision, not the quant type.**
- Q4_K_M: attention/output → Q6_K, FFN → Q4_K
- Q4_K_S: everything → Q4_K
- Q5_K_M: attention/output → Q6_K, FFN → Q5_K

This means a single GGUF file can contain multiple quant types. The dequantizer must handle each tensor according to its actual type, not the file's overall quant label.

### Block Structures
```
Q4_K:  144 bytes / 256 elements  — d(f16) + dmin(f16) + scales[12] + qs[128]
Q5_K:  176 bytes / 256 elements  — d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128]
Q6_K:  210 bytes / 256 elements  — ql[128] + qh[64] + scales[16] + d(f16)
Q8_0:   34 bytes / 32 elements   — d(f16) + qs[32]
Q4_0:   18 bytes / 32 elements   — d(f16) + qs[16]
```

---

## 7. Implications for tinyserve

### What We Have Now
- **GPT-OSS-20B (HF safetensors):** MXFP4 native compute via Triton `dot_scaled` on Blackwell. Already optimal. No GGUF involved.
- **GGUF models:** Broken. Tries to dequant everything to BF16, OOMs on large models.

### Three Viable Approaches

**Approach A: llama.cpp style — custom CUDA kernels per quant type**
- Store native Q4_K/Q5_K/Q6_K in GPU cache
- Write per-type fused dequant+matmul CUDA/Triton kernels
- Maximum performance, zero wasted bandwidth
- **Dev effort: HIGH** (one kernel per quant type, each with shared memory tiling)

**Approach B: ExLlamaV2 style — fused dequant Triton kernel**
- Store native quant bytes in GPU cache
- Write ONE Triton kernel that dequants any K-quant type to FP16 inside the matmul
- Good performance (TC FP16 matmul is fast)
- **Dev effort: MEDIUM** (one parameterized kernel, not per-type)

**Approach C: torch int4pack — convert at cache fill**
- Store INT4 packed in GPU cache (4x smaller than BF16)
- Convert Q4_K → float → INT4 pack when expert enters cache (~1ms one-time per expert)
- Use `torch._weight_int4pack_mm` (tinygemm kernel, optimized for batch=1)
- Simplest, still fast, but lossy Q4K→INT4 conversion
- **Dev effort: LOW** (reuse existing `_float_to_int4pack`, existing `GPUINT4Forward`)

### Recommended Path

**Start with C (int4pack), iterate to B later.**

Rationale:
1. C unblocks Qwen 122B and all GGUF models immediately
2. C is ~3 files changed, reuses existing code
3. The int4pack conversion happens once per cache miss (~1ms), not per token
4. At 90%+ cache hit rate, the conversion cost is amortized to near zero
5. tinygemm at batch=1 is well-optimized for our decode regime
6. B (fused Triton kernel) is a performance optimization we can add later without architecture changes — just swap the forward kernel for cache hits

**The real waste to eliminate is the upfront full-model dequant, not the per-expert-on-miss conversion.**

### Storage Architecture

```
GGUF on SSD (Q4_K/Q5_K/Q6_K native)
    │
    ▼ mmap (zero RAM)
MmapExpertStore — offset table per (layer, expert)
    │
    ▼ cache miss: read Q*_K bytes → float32 → int4pack (~1ms)
GPU ExpertCache — INT4 packed slots (4x more capacity)
    │
    ▼ cache hit: torch._weight_int4pack_mm (tinygemm kernel)
Output tensor
```

Peak RAM: ~2 GB (model skeleton + KV cache). Expert data entirely on SSD via mmap.
GPU cache: 4x more experts than BF16 in same VRAM.
