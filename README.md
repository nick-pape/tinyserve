# tinyserve — MoE Expert Offloading for Consumer GPUs

Run Mixture-of-Experts models that don't fit in VRAM on a single NVIDIA GPU.

tinyserve offloads MoE expert weights to CPU RAM and caches hot experts on GPU. A 20B MoE model needs ~10 GB CPU RAM and runs on 8 GB VRAM.

**If your model works in Ollama or llama.cpp, use those.** Their C++ loop is faster. tinyserve is for models they don't support yet, or when you want readable Python (~7K LOC, no compiled extensions).

## Performance (measured)

All numbers from RTX PRO 2000 8 GB **laptop** GPU, GPT-OSS-20B (MXFP4, 238 cache slots, 24 layers × 32 experts, top_k=4). Raw logs in [`benchmarks/`](benchmarks/).

### Cache hit rates (diverse workloads)

Measured with industry-standard methodology: diverse prompts across 5 domains, cold start, domain shifts. NOT measured on warm cache with repeated text. Policy: LFRU with CPU-on-miss (miss penalty ~2ms vs ~20ms for cold loads).

| Workload | Hit rate | tok/s | Source |
|---|---|---|---|
| Cold start (8 diverse prompts) | 79-94% | 8-13 | `miss_penalty_20260331.json` |
| Sustained code | 85% | 8-11 | same |
| Sustained math | 84% | 8-11 | same |
| Sustained creative | 90% | 10-13 | same |
| Sustained multilingual | 95% | 10-13 | same |
| Sustained conversation | 85% | 7-12 | same |
| Domain shift (creative→math) | 84% (-9.6%) | — | same |
| Domain shift (multilingual→code) | 86% (-9.4%) | — | same |

### Per-layer hit rate

LFRU rebalances cache budget toward deeper layers compared to plain LRU:

| Layers | Hit rate | Note |
|---|---|---|
| 0-10 | 39-68% | Early layers |
| 11-17 | 56-74% | Middle layers |
| 18-23 | 54-69% | Deep layers (was 0-10% with LRU) |

This is with 238 slots across 768 total experts (31% coverage). 7-policy comparison in `benchmarks/comprehensive_20260326.txt`: LFRU wins on deep-layer coverage (52% vs LRU's 8%) at comparable overall HR.

### Decode speed (post-prefill)

When cache is warm from prefill of the SAME prompt, decode speed is ~30 tok/s and constant across context lengths. This represents the ceiling, not typical workload performance.

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

**Baseline:** HuggingFace `device_map="auto"`: 0.19 tok/s.

### What we have NOT measured

- No Ollama/llama.cpp comparison on same GPU (they don't support GPT-OSS-20B)
- Only GPT-OSS-20B benchmarked end-to-end
- GGUF Q4_K loader unit-tested only — not tested on real GGUF files
- No multi-user / batch inference benchmarks
- No H100/A100 numbers — all data is from an 8 GB laptop GPU

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
2. **GPU LFRU cache** — Pre-allocated VRAM tensor. Hit: template params set to cache slot views (zero-copy). Miss: CPU-on-miss path reduces penalty from ~20ms to ~2ms; double-buffered H2D pipeline for async loads.
3. **FATE cross-layer prefetch** — Current layer's hidden states predict next layer's experts. Prefetch overlaps with attention compute.
4. **Temporal routing fallback** — After first token, reuse previous token's routing. Matches or exceeds FATE accuracy.
5. **Batched expert prefill** — Groups tokens by expert, loads each once. Reduces prefill expert loads from O(seq_len × top_k) to O(num_experts).

## Supported models

| Model | Params | RAM needed | Status |
|---|---|---|---|
| GPT-OSS-20B | 20B (MXFP4) | ~10 GB | **Benchmarked** |
| Qwen 3.5 MoE 35B | 35B | ~18 GB | Unit tested |
| Mixtral 8x7B | 47B | ~24 GB | Unit tested |
| GPT-OSS-120B | 120B | ~60 GB | Profile only |
| DeepSeek-V3/R1 | 671B | ~350 GB | Profile only |
| + 6 more families | varies | varies | Profile only |

**Status:** "Benchmarked" = real weights loaded, tokens generated, throughput measured. "Unit tested" = pipeline runs on mock weights. "Profile only" = architecture metadata exists, no code run.

**Formats:** HuggingFace safetensors (BF16, FP8, MXFP4). GGUF (Q4_K/Q5_K/Q6_K) unit-tested.

## Configuration

```python
model = load_and_offload(
    "openai/gpt-oss-20b",
    cache_capacity=0,              # 0 = auto-size from VRAM
    cache_policy="lfru",           # lru, lfru, slru, lfu, fifo, ls, dali
    max_seq_len=4096,              # static KV cache (0 = dynamic)
    gpu_memory_utilization=0.90,
)
```

## Limitations

- NVIDIA only (CUDA streams, Triton PTX)
- Single GPU only
- Batch size 1 decode only (template weight swapping is not batch-safe)
- Deep layers (18-23) had near-zero hit rates with LRU; LFRU brings them to 54-69% at 238 slots
- Prefill is attention-dominated, not improved by expert caching
- GGUF loader not tested on real files

## Benchmarking

```bash
# Industry-standard diverse workload benchmark
python -m scripts.cache_benchmark --model openai/gpt-oss-20b

# Decode speed benchmark
python scripts/benchmark.py --model openai/gpt-oss-20b

# Context scaling (prefill vs decode separated)
python scripts/benchmark.py --context-scaling

# Cache policy comparison
python scripts/benchmark.py --compare-policies

# FATE prediction accuracy
python scripts/benchmark.py --fate-diagnostic
```

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ --ignore=tests/test_hf_models.py -x -q   # 335 tests
```

## License

MIT
