# GGUF-to-HuggingFace Weight Mapping — Ecosystem Approaches

*2026-04-04*

---

## The Answer: `gguf/tensor_mapping.py` IS the Universal Mapping Table

`pip install gguf` provides a 2000-line data-driven mapping table covering 122 architectures and 299 tensor types. Both HuggingFace transformers and vLLM import and use it. It is maintained by the llama.cpp team.

```python
from gguf import get_tensor_name_map
name_map = get_tensor_name_map("qwen3moe", n_blocks=48)
# Maps HF names <-> GGUF canonical names
```

## What Is Data-Driven vs Per-Model Code

| Aspect | Data-Driven? | Location |
|--------|-------------|----------|
| Name mapping (HF <-> GGUF) | **YES** | `gguf/tensor_mapping.py` |
| Config mapping (metadata <-> HF config) | **YES** | `transformers/integrations/ggml.py` |
| Weight transforms (permute, transpose, fuse) | **NO** — per-model code | HF: `TensorProcessor` subclasses (12 models); vLLM: in `load_weights()` |
| Tensor existence per architecture | **YES** | `gguf/constants.py` `MODEL_TENSORS` dict |

## How Projects Handle It

### HuggingFace Transformers
- Uses `gguf.get_tensor_name_map()` for name mapping (reverse direction: GGUF→HF)
- Per-model `TensorProcessor` subclasses for weight transforms:
  - `LlamaTensorProcessor` — reverse Q/K head permutation
  - `Qwen2MoeTensorProcessor` — fused expert gate/up interleaving
  - `MambaTensorProcessor` — conv1d unsqueeze, A-matrix log transform
  - 9 more...

### vLLM
- Same `gguf.get_tensor_name_map()` foundation
- Keeps weights quantized (no dequant during load)
- Weight transforms happen in each model's `load_weights()` method

### llama.cpp / llama-cpp-python
- **Bypasses HF entirely** — reads GGUF natively using canonical names
- No mapping problem because there's no HF model to map to

## The Key Insight for tinyserve

The mapping problem exists ONLY because we use HF's model skeleton (which expects HF tensor names). Two paths forward:

**Path A: Use the gguf-py mapping table + HF TensorProcessor**
- Import `gguf.get_tensor_name_map()` — no custom mapping code
- Copy/adapt HF's `TensorProcessor` for Qwen3.5MoE (handle fused QKV, shared experts, SSM)
- Scales to new models automatically via gguf-py updates

**Path B: Don't use HF model skeleton at all**
- Like llama.cpp: read GGUF, build compute graph directly
- No mapping problem — use canonical GGUF tensor names
- But loses HF's attention implementation, position encoding, etc.

**Path C (hybrid): Use HF for non-MoE layers, ggml for everything else**
- HF handles attention logic (complex, model-specific)
- ggml handles all weight storage and matmul
- Mapping only needed for the HF skeleton's parameter names

## Sources

1. `gguf/tensor_mapping.py` — [ggml-org/ggml](https://github.com/ggml-org/ggml)
2. `transformers/modeling_gguf_pytorch_utils.py` — [huggingface/transformers](https://github.com/huggingface/transformers)
3. `vllm/model_executor/model_loader/gguf_loader.py` — [vllm-project/vllm](https://github.com/vllm-project/vllm)
