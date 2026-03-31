"""Collect expert routing decisions for buddy co-activation profiling."""
import json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from tinyserve.offload import load_and_offload
from tinyserve.buddy_experts import build_coactivation_matrix, BuddyTable
from scripts.prompts import COLD_START, CODE_PROMPTS, MATH_PROMPTS, CREATIVE_PROMPTS, MULTILINGUAL_PROMPTS, CONVERSATION_PROMPTS

LOG = "benchmarks/buddy_calibration_20260331.txt"
def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f: f.write(m + "\n")

def main():
    with open(LOG, "w"): pass
    log("=== Buddy Expert Calibration ===")

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    model = load_and_offload("openai/gpt-oss-20b", attn_implementation="sdpa")

    # Collect routing decisions from diverse prompts
    all_prompts = COLD_START + CODE_PROMPTS + MATH_PROMPTS + CREATIVE_PROMPTS + MULTILINGUAL_PROMPTS + CONVERSATION_PROMPTS
    log(f"Running {len(all_prompts)} prompts to collect routing data...")

    # Hook into the routing to capture expert selections per layer
    from tinyserve.offloaded_model import _last_routing
    routing_per_layer: dict[int, list[list[int]]] = {}

    # Generate tokens and collect routing
    for i, prompt in enumerate(all_prompts):
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        with torch.inference_mode():
            model.generate(**inp, max_new_tokens=20, do_sample=False)

        # Collect routing from _last_routing (populated during decode)
        for layer_idx, routing_tensor in _last_routing.items():
            if layer_idx not in routing_per_layer:
                routing_per_layer[layer_idx] = []
            if isinstance(routing_tensor, torch.Tensor):
                routing_per_layer[layer_idx].append(routing_tensor.tolist())

        if (i + 1) % 10 == 0:
            log(f"  {i+1}/{len(all_prompts)} prompts done")

    log(f"\nCollected routing for {len(routing_per_layer)} layers")

    # Build per-layer co-activation matrices and buddy tables
    num_experts = 32  # GPT-OSS-20B
    buddy_tables = {}
    for layer_idx in sorted(routing_per_layer.keys()):
        decisions = routing_per_layer[layer_idx]
        if not decisions:
            continue
        routing_tensor = torch.tensor(decisions)
        if routing_tensor.dim() == 1:
            routing_tensor = routing_tensor.unsqueeze(0)
        coact = build_coactivation_matrix(routing_tensor, num_experts)
        table = BuddyTable.from_coactivation(coact, max_buddies=3)
        buddy_tables[layer_idx] = table

        # Show top buddies for this layer
        top_pair = coact.triu(diagonal=1).argmax()
        e1, e2 = top_pair // num_experts, top_pair % num_experts
        log(f"  Layer {layer_idx:2d}: top pair E{e1.item()}-E{e2.item()} (coact={coact[e1,e2].item():.0f}), "
            f"buddies of E0={table.get_buddies(0)[:3]}")

    # Save buddy tables as JSON
    output = {}
    for li, table in buddy_tables.items():
        output[str(li)] = {str(eid): buddies for eid, buddies in table._buddies.items()}

    out_path = "benchmarks/buddy_tables_gptoss20b.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nBuddy tables saved to {out_path}")
    log(f"Layers with tables: {len(buddy_tables)}")
    log("Done.")

if __name__ == "__main__":
    main()
