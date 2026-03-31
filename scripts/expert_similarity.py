"""Analyze inter-expert weight similarity for D2-MoE feasibility.

If experts within a layer have high cosine similarity and low-rank deltas
from the mean, D2-MoE compression (shared base + low-rank deltas) is viable.
If not, it's not worth implementing.
"""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG = "benchmarks/expert_similarity_20260331.txt"

def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f: f.write(m + "\n")

def main():
    with open(LOG, "w"): pass
    log("=== Expert Weight Similarity Analysis (D2-MoE feasibility) ===")
    log("Model: openai/gpt-oss-20b (MXFP4, dequanted to BF16 for analysis)")

    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b")
    log(f"Layers: {config.num_hidden_layers}, Experts: {config.num_local_experts}")
    log(f"Hidden: {config.hidden_size}, Intermediate: {config.intermediate_size}")

    # Load model weights on CPU (BF16, dequanted from MXFP4 by HF)
    log("\nLoading model weights on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b", dtype=torch.bfloat16, device_map="cpu"
    )

    log("Analyzing expert weight similarity...\n")
    log(f"{'Layer':>6} {'CosSimMean':>12} {'CosSimMin':>11} {'Rank@90%':>10} {'Rank@95%':>10} {'Rank@99%':>10}")
    log("-" * 65)

    inner = model.model if hasattr(model, "model") else model
    layers = inner.layers

    results = []
    for li in range(min(config.num_hidden_layers, 24)):
        layer = layers[li]
        # Find the MoE block and extract expert weights
        moe = None
        for name, mod in layer.named_modules():
            if hasattr(mod, "experts") and isinstance(getattr(mod, "experts", None), (list, torch.nn.ModuleList)):
                moe = mod
                break
        if moe is None:
            log(f"  Layer {li}: no MoE block found, skipping")
            continue

        # Extract gate_up_proj weights for all experts
        expert_weights = []
        for expert in moe.experts:
            if hasattr(expert, "gate_up_proj"):
                w = expert.gate_up_proj.data.float().flatten()
            elif hasattr(expert, "w1"):
                w = expert.w1.data.float().flatten()
            else:
                break
            expert_weights.append(w)

        if len(expert_weights) < 2:
            log(f"  Layer {li}: <2 experts extracted, skipping")
            continue

        # Stack: [num_experts, flattened_size]
        W = torch.stack(expert_weights)
        n_experts, feat_dim = W.shape

        # 1. Pairwise cosine similarity
        W_norm = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_sim = W_norm @ W_norm.T
        # Exclude diagonal
        mask = ~torch.eye(n_experts, dtype=torch.bool)
        sim_values = cos_sim[mask]
        mean_sim = sim_values.mean().item()
        min_sim = sim_values.min().item()

        # 2. SVD of delta matrix (W - mean)
        W_mean = W.mean(dim=0, keepdim=True)
        delta = W - W_mean  # [n_experts, feat_dim]

        # Compute singular values of delta
        # Use a subset of singular values (full SVD is too expensive for large dims)
        max_rank = min(n_experts, 256)  # cap at 256
        try:
            U, S, Vh = torch.svd_lowrank(delta.float(), q=max_rank)
            total_energy = (S ** 2).sum().item()
            cumulative = (S ** 2).cumsum(dim=0) / total_energy

            rank_90 = (cumulative < 0.90).sum().item() + 1
            rank_95 = (cumulative < 0.95).sum().item() + 1
            rank_99 = (cumulative < 0.99).sum().item() + 1
        except Exception:
            rank_90 = rank_95 = rank_99 = -1

        log(f"{li:>6} {mean_sim:>11.4f} {min_sim:>10.4f} {rank_90:>10} {rank_95:>10} {rank_99:>10}")
        results.append({
            "layer": li, "cos_sim_mean": round(mean_sim, 4), "cos_sim_min": round(min_sim, 4),
            "rank_90": rank_90, "rank_95": rank_95, "rank_99": rank_99,
        })

    # Summary
    if results:
        avg_sim = sum(r["cos_sim_mean"] for r in results) / len(results)
        avg_rank95 = sum(r["rank_95"] for r in results) / len(results)
        log(f"\nAverage cosine similarity: {avg_sim:.4f}")
        log(f"Average rank for 95% energy: {avg_rank95:.1f}")

        if avg_sim > 0.7:
            log("\nVERDICT: HIGH similarity — D2-MoE delta compression is VIABLE.")
            log(f"Expected compression: ~{100*(1 - avg_rank95/feat_dim*n_experts):.0f}% (rank-{avg_rank95:.0f} deltas)")
        elif avg_sim > 0.3:
            log("\nVERDICT: MODERATE similarity — D2-MoE might work but gains are limited.")
        else:
            log("\nVERDICT: LOW similarity — D2-MoE is NOT viable. Experts are too diverse.")

    import json
    with open("benchmarks/expert_similarity_20260331.json", "w") as f:
        json.dump(results, f, indent=2)
    log("\nDone.")

if __name__ == "__main__":
    main()
