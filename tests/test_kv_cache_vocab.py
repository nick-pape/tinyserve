import torch
from tinyserve.kv_cache import KVCache, KVCacheOverflow


def test_kv_cache_class_name():
    cache = KVCache(max_seq_len=16, num_layers=2, num_kv_heads=4,
                    head_dim=8, device=torch.device("cpu"))
    assert isinstance(cache, KVCache)


def test_preallocate_constructor():
    class FakeConfig:
        num_hidden_layers = 2
        num_key_value_heads = 4
        num_attention_heads = 8
        hidden_size = 64

    cache = KVCache.preallocate(FakeConfig(), max_context_tokens=32, device="cpu")
    assert cache.max_seq_len == 32


def test_store_layer_alias_exists():
    cache = KVCache(max_seq_len=16, num_layers=2, num_kv_heads=2,
                    head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16)
    assert hasattr(cache, "store_layer")


def test_overflow_attr_name():
    exc = KVCacheOverflow(overflow_token_count=10)
    assert exc.overflow_token_count == 10


def test_enable_sliding_window():
    cache = KVCache(max_seq_len=64, num_layers=1, num_kv_heads=2,
                    head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16)
    cache.enable_sliding_window(kv_window_tokens=16, kv_sink_tokens=4)
    assert cache._window_size == 16
