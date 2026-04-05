from tinyserve.cache_policy import EvictionPolicy, LRUPolicy, make_eviction_policy

def test_eviction_policy_abc_name():
    assert issubclass(LRUPolicy, EvictionPolicy)

def test_locate_returns_slot_on_hit():
    p = LRUPolicy()
    p.register((0, 1), slot=5)
    assert p.locate((0, 1)) == 5

def test_locate_returns_none_on_miss():
    p = LRUPolicy()
    assert p.locate((0, 99)) is None

def test_make_eviction_policy_factory():
    p = make_eviction_policy("lfru", capacity=32)
    p.register((0, 0), slot=0)
    assert p.locate((0, 0)) == 0
