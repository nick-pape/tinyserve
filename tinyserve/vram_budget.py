"""Dynamic VRAM rebalancing between expert cache and KV cache.

Monitors KV usage and expert hit rate. When KV approaches capacity,
evicts experts to free VRAM for KV extension. When a request completes,
grows expert cache back.

The tradeoff: 1 expert slot ≈ N tokens of KV (model-dependent).
For GPT-OSS-20B: 1 slot = 13.2 MB / 48 KB = 275 tokens.
"""

import logging

logger = logging.getLogger(__name__)


class VRAMBudget:
    """Controller for dynamic expert↔KV VRAM rebalancing."""

    def __init__(
        self,
        expert_cache,
        kv_cache,
        expert_bytes: int,
        kv_bytes_per_token: int,
        max_expert_capacity: int | None = None,
        kv_pressure_threshold: float = 0.85,
        kv_release_threshold: float = 0.10,
        min_expert_capacity: int = 4,
    ):
        self.expert_cache = expert_cache
        self.kv_cache = kv_cache
        self.expert_bytes = expert_bytes
        self.kv_bytes_per_token = kv_bytes_per_token
        self.max_expert_capacity = max_expert_capacity or expert_cache.capacity
        self.kv_pressure_threshold = kv_pressure_threshold
        self.kv_release_threshold = kv_release_threshold
        self.min_expert_capacity = min_expert_capacity
        self.tokens_per_expert_slot = expert_bytes // max(1, kv_bytes_per_token)

    def kv_utilization(self) -> float:
        """Current KV cache utilization (0.0 to 1.0)."""
        if self.kv_cache is None or self.kv_cache.max_seq_len == 0:
            return 0.0
        max_seq = max(self.kv_cache._seq_lens) if self.kv_cache._seq_lens else 0
        return max_seq / self.kv_cache.max_seq_len

    def check(self) -> dict:
        """Check if rebalancing is needed.

        Returns dict with:
            should_rebalance: bool
            direction: "shrink_experts" | "grow_experts" | None
            expert_slots_to_free: int (positive = shrink, negative = grow)
            kv_tokens_gained: int
        """
        kv_util = self.kv_utilization()
        expert_cap = self.expert_cache.capacity

        # KV under pressure → shrink experts
        if kv_util >= self.kv_pressure_threshold and expert_cap > self.min_expert_capacity:
            kv_needed = int(self.kv_cache.max_seq_len * 0.25)
            slots_needed = max(1, kv_needed // max(1, self.tokens_per_expert_slot))
            slots_available = expert_cap - self.min_expert_capacity
            slots_to_free = min(slots_needed, slots_available)
            return {
                "should_rebalance": True,
                "direction": "shrink_experts",
                "expert_slots_to_free": slots_to_free,
                "kv_tokens_gained": slots_to_free * self.tokens_per_expert_slot,
            }

        # KV nearly empty + expert cache below max → grow experts
        if kv_util <= self.kv_release_threshold and expert_cap < self.max_expert_capacity:
            slots_to_grow = self.max_expert_capacity - expert_cap
            return {
                "should_rebalance": True,
                "direction": "grow_experts",
                "expert_slots_to_free": -slots_to_grow,
                "kv_tokens_gained": 0,
            }

        return {
            "should_rebalance": False,
            "direction": None,
            "expert_slots_to_free": 0,
            "kv_tokens_gained": 0,
        }

    def execute(self, action: dict) -> None:
        """Execute a rebalance action returned by check()."""
        if not action["should_rebalance"]:
            return

        if action["direction"] == "shrink_experts":
            n = action["expert_slots_to_free"]
            freed = self.expert_cache.shrink(n)
            kv_tokens = freed // max(1, self.kv_bytes_per_token)
            self.kv_cache.extend(kv_tokens)
            logger.info(
                "Rebalance: freed %d expert slots → +%d KV tokens (cap now %d/%d)",
                n, kv_tokens, self.expert_cache.capacity, self.max_expert_capacity,
            )

        elif action["direction"] == "grow_experts":
            n = -action["expert_slots_to_free"]
            kv_to_shrink = n * self.tokens_per_expert_slot
            max_used = max(self.kv_cache._seq_lens) if self.kv_cache._seq_lens else 0
            available_kv = self.kv_cache.max_seq_len - max_used
            kv_to_shrink = min(kv_to_shrink, available_kv)
            actual_slots = kv_to_shrink // max(1, self.tokens_per_expert_slot)
            if actual_slots > 0:
                self.expert_cache.grow(actual_slots)
                logger.info(
                    "Rebalance: grew expert cache by %d slots (cap now %d/%d)",
                    actual_slots, self.expert_cache.capacity, self.max_expert_capacity,
                )
