# tinyserve — MoE Expert Offloading for Consumer GPUs

Run Mixture-of-Experts models that don't fit in VRAM on a single NVIDIA GPU.

tinyserve offloads MoE expert weights to CPU RAM and caches hot experts on GPU. A 20B MoE model needs ~10 GB CPU RAM and runs on 8 GB VRAM.

**If your model works in Ollama or llama.cpp, use those.** Their C++ loop is faster. tinyserve is for models they don't support yet, or when you want readable Python (~7K LOC, no compiled extensions).

## Performance (measured)

All numbers from a single hardware configuration: **RTX PRO 2000 8 GB laptop GPU**, GPT-OSS-20B (MXFP4, 238 cache slots, 24 layers × 32 experts, top_k=4). No other hardware or model has been benchmarked end-to-end. Raw logs in [`benchmarks/`](benchmarks/).

### Cache hit rates (diverse workloads)

Measured on diverse prompts across 5 domains (code, math, creative, multilingual, conversation) plus domain shifts. Policy: LFRU (frequency-recency hybrid). CPU expert compute on cache miss (~2ms penalty via OneDNN AMX/AVX, instead of ~20ms H2D weight transfer).

**Note on methodology:** "Sequential diverse prompts" resets hit/miss counters but not cache contents between prompts — earlier prompts warm the cache for later ones. True cold-start (empty cache) hit rate for the first prompt is lower (~86-89%). The sustained-domain numbers reflect realistic within-session behavior.

| Workload | Hit rate | tok/s | Source |
|---|---|---|---|
| Sequential diverse prompts (8 prompts) | 86-95% | 9-13 | `cpu_slotmap_bench_20260331` |
| Sustained code (4 prompts) | 93% | 10-12 | same |
| Sustained math (4 prompts) | 93% | 10-11 | same |
| Sustained creative (4 prompts) | 95% | 11-12 | same |
| Sustained multilingual (4 prompts) | 95% | 11-12 | same |
| Sustained conversation (4 prompts) | 95% | 8-12 | same |
| Domain shift (creative→math) | 94% (-1.8%) | — | same |
| Domain shift (multilingual→code) | 94% (-1.6%) | — | same |

**Sample size caveat:** 4 prompts × 30 tokens per domain. Per-prompt tok/s variance is ~15-20% CV. These are point estimates, not statistically tight bounds.

### Per-layer hit rate

With LRU, early layers monopolize the cache (they execute first every forward pass and always "look recent"). LFRU's frequency component rebalances this:

| Layers | LFRU hit rate | LRU hit rate | Source |
|---|---|---|---|
| 0-10 | 84-97% | 28-41% | `cpu_slotmap_bench` / `comprehensive_20260326` |
| 11-17 | 89-96% | 11-32% | same |
| 18-23 | 79-94% | 0-10% | same |

This is with 238 slots across 768 total experts (31% coverage). LRU per-layer starvation is a structural property of sequential layer execution, not model-specific.

### Expert frequency distribution (GPT-OSS-20B)

Expert activation is relatively **flat**, not Zipf-like:
- Top 10% of experts handle 44% of accesses
- Top 25% handle 75%
- 73% of all 768 experts activated across a diverse workload

Implication: frequency-based pinning (keeping "hot" experts) has limited value because the hot set is large. Dynamic policies (LFRU, SLRU) work better than static approaches for this model.

### Decode speed (post-prefill, warm cache)

When cache is warm from prefill of the SAME prompt, decode speed is ~30 tok/s and constant across context lengths. **This represents the ceiling, not typical workload performance.** Real-world tok/s (10-13) is lower due to cache misses across diverse prompts.

| Context | Decode tok/s | Post-prefill HR | Source |
|---|---|---|---|
| 10 | 28.8 | 97% | `batched_prefill_20260326.txt` |
| 100 | 32.2 | 100% | same |
| 1,000 | 30.0 | 100% | same |
| 2,000 | 27.2 | 99% | same |

### End-to-end (prefill + decode)

| Context | Prefill | Total tok/s |
|---|---|---|
| 10 | 1.1s | 11.4 |
| 100 | 1.8s | 8.4 |
| 500 | 2.5s | 6.3 |
| 1,000 | 3.2s | 5.2 |

### Performance ceiling analysis

At 10-13 tok/s, we're at **~36% of the realistic ceiling** (~32 tok/s) for this hardware. The gap is:
- ~60% Python interpreter overhead (layer loop, dict ops, torch dispatch)
- ~15% CUDA synchronization and kernel launch
- ~15% PCIe miss transfers + CPU fallback compute
- ~10% HF generate framework tax

A C++ forward loop would be the single highest-ROI improvement (~2x), but is a major rewrite. All algorithmic/caching/scheduling optimizations from a 28-paper research survey have been either implemented or analyzed and ruled out for this specific model/hardware combination.

### What we have NOT measured

- No Ollama/llama.cpp comparison on same GPU (they don't support GPT-OSS-20B natively)
- Only GPT-OSS-20B benchmarked end-to-end — other models have only architecture profiles
- HuggingFace baseline (0.19 tok/s) was measured once; no benchmark file backs this specific number
- GGUF Q4_K loader unit-tested only — GGUF parsing of Qwen 122B verified but no inference run
- No multi-user / batch inference benchmarks (system is batch=1 only)
- No H100/A100 numbers — all data is from one 8 GB laptop GPU
- No confidence intervals — sample sizes are 4 prompts × 30 tokens per workload

### What we tried and ruled out

| Technique | Result | Why |
|---|---|---|
| D2-MoE delta compression | NOT viable | Expert cosine similarity = 0.0006 (independent weights) |
| Cache bias routing (0.0-3.0) | No effect | Routing already stable for this model |
| Cython hot path | 3.4x microbench, 0% e2e | Python dict loops are <1% of total time |
| GPU INT4 on 8GB | OOMs | Conversion cache exceeds VRAM |
| Expert deferral | Garbage output | KTransformers pattern doesn't work here |
| FlexAttention default | VRAM bug | pytorch #155065, 3-67x more VRAM than SDPA |

## Quick start

```bash
git clone https://github.com/e1n00r/tinyserve.git
cd tinyserve
pip install -e "."
```

```python
from tinyserve import load_and_offload

model = load_and_offload("openai/gpt-oss-20b")
output = model.generate(input_ids, max_new_tokens=100)
```

### CLI

```bash
tinyserve serve --model openai/gpt-oss-20b --port 8000   # OpenAI-compatible HTTP
tinyserve run --model openai/gpt-oss-20b                  # Interactive REPL
tinyserve info --model openai/gpt-oss-20b                 # Model profile
```

### Docker

```bash
docker build -t tinyserve .
docker run --gpus all -p 8000:8000 tinyserve
```

## How it works

1. **Expert store** — Weights packed as flat byte buffers in pinned CPU memory. MXFP4 loaded as raw uint8 blocks + scales from safetensors (no dequantization).
2. **GPU LFRU cache** — Pre-allocated VRAM tensor with frequency-recency eviction. Hit: zero-copy MXFP4 inline forward via Triton `dot_scaled`. Miss: CPU expert compute via OneDNN (~2ms) instead of H2D weight transfer (~20ms).
3. **CPU slot map** — Cache tracking on CPU numpy array (written per-allocate, zero CUDA overhead). GPU tensor synced lazily before lookup. Eliminated 1466 CUDA kernel launches/token.
4. **FATE cross-layer prefetch** — Current layer's hidden states predict next layer's experts. Prefetch overlaps with attention compute.
5. **Temporal routing fallback** — After first token, reuse previous token's routing decisions.
6. **Batched expert prefill** — Groups tokens by expert, loads each once. O(num_experts) vs O(seq_len × top_k).
7. **Buddy expert substitution** — On cache miss, substitute a co-activation-similar cached expert (zero stall). Falls back to CPU compute if no buddy available.

## Supported models

| Model | Params | RAM needed | Status |
|---|---|---|---|
| GPT-OSS-20B | 20B (MXFP4) | ~10 GB | **Benchmarked** |
| Qwen 3.5 MoE 35B | 35B | ~18 GB | Unit tested |
| Mixtral 8x7B | 47B | ~24 GB | Unit tested |
| GPT-OSS-120B | 120B | ~60 GB | Profile only |
| DeepSeek-V3/R1 | 671B | ~350 GB | Profile only |
| + 6 more families | varies | varies | Profile only |

**Status:** "Benchmarked" = real weights loaded, tokens generated, throughput measured on one hardware config. "Unit tested" = pipeline runs on mock weights shaped like that architecture. "Profile only" = architecture metadata exists, no code run.

**Formats:** HuggingFace safetensors (BF16, FP8, MXFP4). GGUF (Q4_K/Q5_K/Q6_K) parsing verified on Qwen 122B, inference not tested.

## Configuration

```python
model = load_and_offload(
    "openai/gpt-oss-20b",
    cache_capacity=0,              # 0 = auto-size from VRAM
    cache_policy="lfru",           # lru, lfru, slru, lfu, fifo, ls, dali
    max_seq_len=4096,              # static KV cache (0 = dynamic)
    gpu_memory_utilization=0.90,
    buddy_table_path="benchmarks/buddy_tables_gptoss20b.json",  # optional
)
```

## Limitations

- **NVIDIA only** — hardcoded CUDA streams and Triton PTX
- **Single GPU only** — no multi-GPU or tensor parallelism
- **Batch size 1 decode only** — template weight swapping is not batch-safe
- **One model benchmarked** — all performance claims are from GPT-OSS-20B on one GPU
- **~36% of theoretical ceiling** — Python overhead dominates; C++ rewrite needed for >20 tok/s
- **No GGUF inference** — GGUF parsing works but end-to-end generation not tested
- **Expert independence** — D2-MoE compression is not viable for GPT-OSS (cosim=0.0006)

## Benchmarking

```bash
# Industry-standard diverse workload benchmark
python -m scripts.cache_benchmark --model openai/gpt-oss-20b

# Cache policy comparison (7 policies)
python scripts/comprehensive_bench.py

# Cache bias sweep
python scripts/sweep_cache_bias.py

# Expert similarity analysis (D2-MoE feasibility)
python scripts/expert_similarity.py

# Buddy co-activation calibration
python scripts/calibrate_buddies.py

# Decode speed benchmark
python scripts/benchmark.py --context-scaling
```

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ --ignore=tests/test_hf_models.py -x -q   # ~340 tests
```

## License

MIT
