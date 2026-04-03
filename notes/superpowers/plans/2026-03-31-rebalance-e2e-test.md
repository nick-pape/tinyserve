# Dynamic VRAM Rebalancing E2E Test

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify that dynamic VRAM rebalancing works end-to-end: KV cache auto-extends by evicting experts when context grows, expert cache grows back between requests.

**Architecture:** Load GPT-OSS-20B with a small initial KV cache (512 tokens). Generate with progressively longer context. Observe expert cache shrinking and KV cache growing. Reset and observe expert cache restoring.

---

### Task 1: Fix the test script and run

**Files:**
- Modify: `scripts/test_dynamic_rebalance.py`

The current script uses `model()` calls with `past_key_values=kv` but the rebalancing is now demand-driven (KV self-heals via `_vram_budget`). The script needs to:

1. Use `model.generate()` with `past_key_values=kv` to trigger real KV updates
2. NOT manually call `budget.check()`/`budget.execute()` — the KV cache self-heals
3. Log expert cache capacity and KV max_seq_len after each generation
4. Generate with a long prompt that will trigger overflow

- [ ] **Step 1: Rewrite the test script**

```python
# Key flow:
# 1. Load with max_seq_len=256 (very small — forces overflow quickly)
# 2. Generate 30 tokens → fills ~30 of 256 KV → no rebalance
# 3. Feed 200-token prompt → fills ~230 of 256 KV → tight
# 4. Feed 300-token prompt → overflow → VRAMBudget fires → expert cache shrinks → KV extends
# 5. Observe expert cache capacity < initial
# 6. Reset KV → call budget.release_kv() → expert cache grows back
# 7. Observe expert cache capacity == initial
```

- [ ] **Step 2: Run as background process**
- [ ] **Step 3: Verify the three phases (stable → shrink → restore)**
- [ ] **Step 4: Commit results**
