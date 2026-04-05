import torch
from tinyserve.kv_cache import KVCache
from tinyserve.vram_budget import VRAMBudget

def test_reclaim_slots_for_kv_name():
    class FakeCache:
        capacity = 10
        def shrink(self, n): return n * 64
        def extend(self, n): pass
    kv = KVCache(max_seq_len=16, num_layers=1, num_kv_heads=1, head_dim=4, device=torch.device("cpu"))
    budget = VRAMBudget(FakeCache(), kv, expert_bytes=64, kv_bytes_per_token=16)
    assert hasattr(budget, "reclaim_slots_for_kv")
