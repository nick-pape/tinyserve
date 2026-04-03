# Quantization Format Support Plan

## Goal
Support all popular quantization formats so users on any hardware with any model
can use tinyserve. Expert offloading is format-agnostic — the cache stores raw
bytes and the kernel handles the math.

## Formats by Priority

### Tier 1 — Already Working / Easy
| Format | Models | Status | Notes |
|--------|--------|--------|-------|
| BF16 / FP16 | All HF models | ✅ Working | Default fallback |
| MXFP4 | GPT-OSS-20B/120B | ⚠️ Partial | from_safetensors path exists; GPU swizzle OOMs on 8GB during load |
| AWQ (4-bit) | Qwen, Llama, Mistral | ✅ Works via HF | AutoAWQ loads transparently; expert extraction works |

### Tier 2 — High Impact, Needs Work
| Format | Models | Approach |
|--------|--------|----------|
| GGUF Q4_K_M / Q5_K_M | Everything | Load via llama-cpp-python; extract expert tensors as float16 for offload. Majority of localllama users use GGUF. |
| GPTQ (4-bit) | Llama, Mistral, Qwen | AutoGPTQ or HF quantization; expert dequant to fp16 for template forward |
| EXL2 (4-bit) | Llama, Mistral | exllamav2 integration; very popular for quality at 4-bit |
| FP8 (e4m3) | vLLM-quantized models | Direct fp8 expert storage; A100/H100 native, Blackwell via emulation |

### Tier 3 — Future / Specialized
| Format | Use Case |
|--------|----------|
| MXFP4 native (no swizzle) | Load raw int4 blocks from safetensors, dequant on-the-fly — bypass HF loader entirely |
| INT4 block-quant (GGUF IQ4_XS etc.) | Importance-matrix quantized; best quality/size for CPU offload |
| 2-bit (GGUF Q2_K) | Extreme memory pressure; cache-miss expert loading from SSD |

## GGUF Integration Plan (Highest Priority Tier 2)

GGUF is the dominant format for local inference (~80% of localllama users).
llama-cpp-python provides Python bindings that can load GGUF models.

### Approach A: llama-cpp-python backend
```python
from llama_cpp import Llama
# Load GGUF, extract expert weight tensors layer by layer
# Convert to fp16 numpy → torch, store in GenericExpertStore
# Run forward via our template (dequant overhead per token)
```
Downside: llama-cpp-python's expert layout is internal; tensor extraction is fragile.

### Approach B: GGUF-to-safetensors conversion (recommended)
Convert GGUF to HF safetensors once at load time using `gguf` Python library:
```python
import gguf
# Read GGUF tensors directly, reconstruct expert weight dict
# Build GenericExpertStore from the dequantized tensors
# Store as fp16 in CPU pinned memory
```
This keeps our pipeline format-agnostic. The GGUF reader is pure Python,
no llama.cpp dependency.

### Key GGUF tensor naming for MoE
```
# Mixtral/Qwen GGUF expert tensor names:
blk.{layer}.ffn_gate_exps.weight   # [n_experts, ffn_dim, hidden_dim]
blk.{layer}.ffn_up_exps.weight     # [n_experts, ffn_dim, hidden_dim]
blk.{layer}.ffn_down_exps.weight   # [n_experts, hidden_dim, ffn_dim]
```

## Expert Size by Format (per expert, Qwen3.5-35B-A3B, 128 experts)

| Format | Bits/weight | Expert size | Cache slots (3.5 GB) |
|--------|------------|-------------|----------------------|
| BF16   | 16         | ~50 MB      | ~70 |
| FP8    | 8          | ~25 MB      | ~140 |
| GPTQ/AWQ/EXL2 | 4 | ~12 MB    | ~290 |
| MXFP4  | 4          | ~12 MB      | ~290 |
| GGUF Q4_K_M | ~4.5 | ~14 MB   | ~250 |
| GGUF Q2_K  | ~2.6  | ~8 MB    | ~440 |

## Implementation Order
1. **GGUF reader** — `src/gguf_store.py`: read expert tensors from .gguf file,
   dequantize to fp16, build GenericExpertStore. No new kernel needed.
2. **FP8 expert store** — store experts in fp8, dequant in template forward.
   Halves PCIe bandwidth vs BF16.
3. **AWQ/GPTQ passthrough** — already works via HF loader; document + test.
4. **MXFP4 bypass loader** — load raw MXFP4 blocks without HF swizzle, use
   existing triton_dequant kernel. Fixes GPT-OSS-20B on 8GB GPU.
5. **EXL2 reader** — exllamav2 tensor format, popular for quality.
