# Backward-compat shim — kept permanently (external API surface)
from .kv_cache import KVCache as StaticKVCache, KVCache, KVCacheOverflow  # noqa: F401
