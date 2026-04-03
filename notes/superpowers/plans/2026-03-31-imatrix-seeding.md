# imatrix-Based Expert Cache Seeding from GGUF

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse imatrix activation counts from GGUF files and use them to pre-seed the expert cache at load time, eliminating the 80-160 token cold-start phase where hit rate is 48-56%.

**Architecture:** Extract per-expert activation counts from imatrix `.dat` files (or GGUF metadata if embedded). Rank experts by activation count per layer. Pre-load the top-N into GPU cache before the first forward pass. The pluggable cache policy then handles runtime eviction.

**Tech Stack:** Existing `tinyserve/gguf_reader.py`, `tinyserve/generic_store.py`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `tinyserve/imatrix.py` | Create | Parse imatrix data, rank experts per layer |
| `tinyserve/generic_store.py` | Modify | Add `seed_cache(layer_expert_ranking)` method |
| `tinyserve/offload.py` | Modify | Add `imatrix_path` parameter, call seeding |
| `tests/test_imatrix.py` | Create | Tests for parsing and seeding |

---

### Task 1: imatrix parser

**Files:**
- Create: `tinyserve/imatrix.py`
- Test: `tests/test_imatrix.py`

imatrix files contain per-tensor activation statistics. For MoE models, expert tensors have `.counts` (number of times activated) and `.in_sum2` (sum of squared activations). We want `.counts` to rank experts.

- [ ] **Step 1: Research imatrix format**

The imatrix `.dat` file format (from llama.cpp `imatrix.cpp`):
- Header: `n_entries` (int32)
- For each entry: `name_len` (int32), `name` (chars), `ncall` (int32), `nval` (int32), `values` (float32 × nval)

Expert tensor names follow: `blk.{L}.ffn_gate.{E}.weight` or `blk.{L}.ffn_gate_exps.weight`

Parse `ncall` per expert tensor → expert activation count per layer.

- [ ] **Step 2: Write tests**

```python
def test_parse_imatrix_dat():
    from tinyserve.imatrix import parse_imatrix
    # Create a synthetic imatrix .dat file
    ...

def test_rank_experts_per_layer():
    from tinyserve.imatrix import rank_experts
    counts = {(0, 0): 100, (0, 1): 50, (0, 2): 200, (0, 3): 10}
    ranking = rank_experts(counts, num_layers=1, num_experts=4)
    assert ranking[0] == [2, 0, 1, 3]  # sorted by count descending

def test_seed_cache_from_ranking():
    # Create cache, seed with top-N experts, verify they're cached
    ...
```

- [ ] **Step 3: Implement parser**
- [ ] **Step 4: Implement cache seeding**
- [ ] **Step 5: Wire into offload.py** (`imatrix_path` parameter)
- [ ] **Step 6: Test with Qwen 122B GGUF** (which has imatrix data)
- [ ] **Step 7: Commit**

---

### Task 2: Test on Qwen 122B

- [ ] **Step 1: Check if Qwen GGUF has imatrix data**
- [ ] **Step 2: If yes, parse and display expert activation ranking**
- [ ] **Step 3: If no, generate imatrix data or use a model that has it**
