#!/usr/bin/env python3
"""Auto-tune expert cache vs KV cache trade-off.

Computes the Pareto frontier analytically from VRAM budget, then
validates selected configs with actual benchmarks.

Usage:
    python scripts/autotune.py --model openai/gpt-oss-20b
    python scripts/autotune.py --model openai/gpt-oss-20b --kv-fp8
    python scripts/autotune.py --model openai/gpt-oss-20b --gpu-memory-utilization 0.85
    python scripts/autotune.py --model openai/gpt-oss-20b --validate  # run benchmarks
"""

import argparse
import sys

import torch

sys.path.insert(0, ".")


def compute_pareto_table(
    total_vram: int,
    model_weights_bytes: int,
    expert_bytes: int,
    kv_bytes_per_token: int,
    gpu_memory_utilization: float = 0.90,
    headroom_bytes: int = 256 * 1024 * 1024,
):
    """Compute expert/KV trade-off table analytically (no GPU needed)."""
    usable = int(total_vram * gpu_memory_utilization)
    available = usable - model_weights_bytes - headroom_bytes
    if available <= 0:
        return []

    seq_lens = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    results = []
    for seq_len in seq_lens:
        kv_bytes = seq_len * kv_bytes_per_token
        expert_budget = available - kv_bytes
        if expert_budget < expert_bytes:
            continue
        expert_slots = expert_budget // expert_bytes
        kv_mb = kv_bytes / 1e6
        expert_mb = expert_slots * expert_bytes / 1e6
        results.append({
            "seq_len": seq_len,
            "kv_mb": kv_mb,
            "expert_slots": expert_slots,
            "expert_mb": expert_mb,
            "total_mb": (kv_bytes + expert_slots * expert_bytes) / 1e6,
        })

    # Mark Pareto-optimal (can't improve context without losing expert capacity)
    for r in results:
        r["pareto"] = True
    # All are Pareto by construction (monotonic trade-off)
    return results


def main():
    parser = argparse.ArgumentParser(description="Auto-tune expert/KV cache trade-off")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--kv-fp8", action="store_true", help="FP8 KV cache (2x context)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use (default 0.90, like vLLM)")
    parser.add_argument("--validate", action="store_true",
                        help="Run actual benchmarks on selected configs")
    parser.add_argument("--gen-tokens", type=int, default=30)
    parser.add_argument("--cache-policy", default="lfru")
    args = parser.parse_args()

    from transformers import AutoConfig

    from tinyserve.static_kv_cache import StaticKVCache

    config = AutoConfig.from_pretrained(args.model)
    effective = getattr(config, "text_config", config)
    kv_dtype = torch.float8_e4m3fn if args.kv_fp8 else torch.bfloat16

    kv_bpt = StaticKVCache.bytes_per_token(
        effective.num_hidden_layers, effective.num_key_value_heads,
        effective.head_dim, kv_dtype,
    )

    # Get GPU info
    total_vram = torch.cuda.get_device_properties(0).total_memory
    gpu_name = torch.cuda.get_device_name(0)

    # Estimate model weight size (non-expert) — load model to measure
    # For now use a reasonable estimate based on model config
    hidden = effective.hidden_size
    n_layers = effective.num_hidden_layers
    n_heads = effective.num_attention_heads
    vocab = effective.vocab_size
    # Attention: Q,K,V,O projections per layer + norms + embeddings
    attn_bytes_per_layer = 4 * hidden * hidden * 2  # BF16
    norm_bytes_per_layer = 2 * hidden * 2
    embed_bytes = vocab * hidden * 2 + hidden * 2  # embed + lm_head + final norm
    model_weights_bytes = n_layers * (attn_bytes_per_layer + norm_bytes_per_layer) + embed_bytes
    # Router weights
    n_experts = effective.num_local_experts
    router_bytes = n_layers * hidden * n_experts * 2
    model_weights_bytes += router_bytes
    # Double-buffer for expert pipeline
    expert_bytes_est = kv_bpt  # rough — will be corrected after load

    print(f"GPU: {gpu_name} ({total_vram / 1e9:.1f} GB)")
    print(f"GPU memory utilization: {args.gpu_memory_utilization:.0%}")
    print(f"KV dtype: {kv_dtype} ({kv_bpt} bytes/token)")
    print(f"Estimated model weights: {model_weights_bytes / 1e9:.2f} GB")
    print()

    # We need actual expert_bytes from the store. Load model once to get it.
    if args.validate:
        from transformers import AutoTokenizer

        from tinyserve.offload import load_and_offload

        print("Loading model for validation...")
        model = load_and_offload(
            args.model, cache_capacity=0, cache_policy=args.cache_policy,
            max_seq_len=0, kv_dtype=kv_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        store = model._offload_pipelines[0].store
        expert_bytes = store.buffer_expert_bytes
        expert_slots_baseline = model._offload_pipelines[0].cache.capacity if model._offload_pipelines[0].cache else 0

        # Measure actual VRAM used by model weights
        used_vram = total_vram - torch.cuda.mem_get_info(0)[0]
        # Subtract expert cache to get pure model weight footprint
        model_weights_bytes = used_vram - expert_slots_baseline * expert_bytes

        print(f"Actual model weights: {model_weights_bytes / 1e9:.2f} GB")
        print(f"Expert size: {expert_bytes / 1e6:.1f} MB")
        print(f"Baseline: {expert_slots_baseline} expert slots (no KV cache)")
        print()

        del model
        torch.cuda.empty_cache()
    else:
        # Use estimates
        expert_bytes = 13_236_480  # MXFP4 default for GPT-OSS-20B

    # Compute analytical table
    table = compute_pareto_table(
        total_vram, model_weights_bytes, expert_bytes,
        kv_bpt, args.gpu_memory_utilization,
    )

    print(f"{'max_seq_len':>12s} {'KV MB':>8s} {'Expert slots':>13s} {'Expert MB':>10s}")
    print("─" * 48)
    for r in table:
        print(f"{r['seq_len']:>12,d} {r['kv_mb']:>7.0f}M {r['expert_slots']:>13d} {r['expert_mb']:>9.0f}M")

    print()
    print("All configs are Pareto-optimal (monotonic trade-off).")
    print("Choose based on your workload:")
    print("  - Short context, max throughput → max expert slots (top rows)")
    print("  - Long context, lower throughput → max seq_len (bottom rows)")

    if args.validate and len(table) > 0:
        # Validate 3 key configs: no-KV, mid, max-context
        pick_indices = [0, len(table) // 2, -1]
        picks = [table[i] for i in pick_indices if i < len(table)]

        print()
        print("=== VALIDATION BENCHMARKS ===")
        print(f"{'Config':>20s} {'tok/s':>8s} {'HR%':>6s}")
        print("─" * 38)

        for cfg in picks:
            seq_len = cfg["seq_len"]
            torch.cuda.empty_cache()
            model = load_and_offload(
                args.model, cache_capacity=0, cache_policy=args.cache_policy,
                max_seq_len=seq_len, kv_dtype=kv_dtype,
            )
            kv = getattr(model, "_kv_cache", None)

            import time

            inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
            # Warmup
            if kv:
                kv.reset()
            with torch.no_grad():
                kw = {"past_key_values": kv} if kv else {}
                model.generate(**inputs, max_new_tokens=5, do_sample=False, **kw)
            # Measure
            if kv:
                kv.reset()
            for p in model._offload_pipelines:
                if p.cache:
                    p.cache.reset_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                kw = {"past_key_values": kv} if kv else {}
                out = model.generate(**inputs, max_new_tokens=args.gen_tokens, do_sample=False, **kw)
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            n = out.shape[1] - inputs["input_ids"].shape[1]
            hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
            misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
            hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
            label = f"{seq_len:,d} ctx"
            print(f"{label:>20s} {n / el:>7.1f} {hr:>5.0f}%")

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
