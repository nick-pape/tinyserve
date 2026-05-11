"""Paged KV cache for multi-request serving.

Manages KV storage in fixed-size pages (blocks of tokens), enabling
multiple concurrent requests to share a single VRAM pool without
pre-allocating max_seq_len per request.

Usage:
    pool = PagedKVPool(num_pages=64, num_layers=24, num_kv_heads=8,
                       head_dim=64, device="cuda", dtype=torch.bfloat16)
    cache = PagedRequestKVCache(pool)
    output = model(input_ids, past_key_values=cache)
    cache.free()  # return pages to pool
"""

import torch

PAGE_SIZE = 256


class PagedKVPool:
    """Pool of pre-allocated KV pages on GPU.

    All pages live in a single contiguous tensor. Individual pages are
    addressed by page_id (index into the first dimension).
    """

    def __init__(
        self,
        num_pages: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_pages = num_pages
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._dtype = dtype
        self._compute_dtype = torch.bfloat16
        self._device = torch.device(device)
        # Shape: [num_pages, num_layers, 2(K/V), num_kv_heads, PAGE_SIZE, head_dim]
        self._pool = torch.zeros(
            num_pages,
            num_layers,
            2,
            num_kv_heads,
            PAGE_SIZE,
            head_dim,
            dtype=dtype,
            device=self._device,
        )
        self._free_pages: list[int] = list(range(num_pages))

    @property
    def pages_free(self) -> int:
        return len(self._free_pages)

    @property
    def pages_used(self) -> int:
        return self.num_pages - len(self._free_pages)

    @property
    def vram_bytes(self) -> int:
        return self._pool.nelement() * self._pool.element_size()

    def allocate_page(self) -> int:
        if not self._free_pages:
            raise RuntimeError("PagedKVPool exhausted: no free pages available")
        return self._free_pages.pop()

    def free_page(self, page_id: int):
        self._free_pages.append(page_id)

    def write(self, page_id: int, layer_idx: int, kv_type: int, offset: int, data: torch.Tensor):
        """Write tokens into a page.

        Args:
            page_id: which page
            layer_idx: which layer
            kv_type: 0=key, 1=value
            offset: token offset within the page
            data: [num_kv_heads, num_tokens, head_dim]
        """
        n = data.shape[-2]
        if self._dtype != self._compute_dtype:
            data = data.to(self._dtype)
        self._pool[page_id, layer_idx, kv_type, :, offset : offset + n] = data

    def read(self, page_ids: list[int], layer_idx: int, kv_type: int, total_tokens: int) -> torch.Tensor:
        """Gather tokens from multiple pages into a contiguous tensor.

        Returns: [1, num_kv_heads, total_tokens, head_dim]
        """
        if not page_ids:
            return torch.zeros(
                1,
                self.num_kv_heads,
                0,
                self.head_dim,
                dtype=self._compute_dtype,
                device=self._device,
            )
        full_pages = total_tokens // PAGE_SIZE
        remainder = total_tokens % PAGE_SIZE

        parts = []
        for i, pid in enumerate(page_ids):
            if i < full_pages:
                parts.append(self._pool[pid, layer_idx, kv_type])
            else:
                parts.append(self._pool[pid, layer_idx, kv_type, :, :remainder])

        # Each part: [num_kv_heads, tokens, head_dim]
        result = torch.cat(parts, dim=1).unsqueeze(0)  # [1, heads, tokens, dim]
        if self._dtype != self._compute_dtype:
            result = result.to(self._compute_dtype)
        return result


class PagedRequestKVCache:
    """Per-request view into a PagedKVPool.

    Lightweight object holding only page_id references. Implements the
    same duck-type interface as StaticKVCache so it can be passed as
    past_key_values to HuggingFace models.
    """

    def __init__(self, pool: PagedKVPool):
        self.pool = pool
        self.page_ids: list[int] = []
        # Per-layer sequence lengths. transformers 5.x's cache contract calls
        # update() once per layer within a single model.forward(). A single
        # global counter would advance with every layer call inside one
        # forward pass, causing layer N to write to position N*new_tokens
        # instead of [0:new_tokens]. The KV reads then return a tensor with
        # k_len = (num_layers-layer_idx) * new_tokens of mostly-zero data,
        # which mismatches the attention mask. Mirror StaticKVCache's
        # per-layer _seq_lens[] design (see static_kv_cache.py:78).
        self._seq_lens: list[int] = [0] * pool.num_layers
        self.is_sliding = [False] * pool.num_layers
        # DeltaNet (linear-attention) state per layer. Hybrid models like
        # Qwen3.5MoE alternate transformer attention layers (use KV cache)
        # with GatedDeltaNet layers (use conv_state + recurrent_state). The
        # DeltaNet path doesn't go through update(); it calls update_conv_state
        # and update_recurrent_state, and reads via self.layers[i].conv_states
        # and .recurrent_states. We keep one slot per layer; for transformer
        # layers these stay None and are ignored.
        from types import SimpleNamespace
        self.layers = [
            SimpleNamespace(conv_states=None, recurrent_states=None)
            for _ in range(pool.num_layers)
        ]

    @property
    def seq_len(self) -> int:
        """Backwards-compat read for any code path that wants a single number.

        Returns the max across layers — in steady state all layers are in
        lockstep, so this matches what the old global `seq_len` would have been.
        """
        return max(self._seq_lens) if self._seq_lens else 0

    def _pages_needed(self, seq_len: int) -> int:
        return (seq_len + PAGE_SIZE - 1) // PAGE_SIZE

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Write new KV data and return full K,V for attention.

        Args:
            key_states: [batch, num_kv_heads, new_tokens, head_dim]
            value_states: same shape
            layer_idx: layer index
            cache_kwargs: dict with optional 'cache_position'

        Returns:
            (k_out, v_out) covering all tokens so far for THIS layer.
        """
        new_tokens = key_states.shape[2]

        if cache_kwargs and "cache_position" in cache_kwargs:
            pos = cache_kwargs["cache_position"]
            start = pos[0].item() if hasattr(pos, "item") else int(pos[0])
        else:
            start = self._seq_lens[layer_idx]

        end = start + new_tokens

        # Page allocation is shared across layers (the pool stores all layers
        # in the same page tensor). Allocate based on the max we'll need.
        pages_needed = self._pages_needed(end)
        while len(self.page_ids) < pages_needed:
            self.page_ids.append(self.pool.allocate_page())

        # Write tokens into pages for THIS layer at [start:end].
        written = 0
        pos_cursor = start
        while written < new_tokens:
            page_idx = pos_cursor // PAGE_SIZE
            offset = pos_cursor % PAGE_SIZE
            can_write = min(PAGE_SIZE - offset, new_tokens - written)

            k_slice = key_states[:, :, written : written + can_write]  # [1, heads, n, dim]
            v_slice = value_states[:, :, written : written + can_write]

            self.pool.write(self.page_ids[page_idx], layer_idx, 0, offset, k_slice.squeeze(0))
            self.pool.write(self.page_ids[page_idx], layer_idx, 1, offset, v_slice.squeeze(0))

            written += can_write
            pos_cursor += can_write

        self._seq_lens[layer_idx] = end

        k_out = self.pool.read(self.page_ids, layer_idx, 0, end)
        v_out = self.pool.read(self.page_ids, layer_idx, 1, end)
        return k_out, v_out

    def get_seq_length(self, layer_idx=0):
        return self._seq_lens[layer_idx]

    def get_max_cache_shape(self):
        return None

    def get_mask_sizes(self, cache_position=None, layer_idx=0):
        # transformers 5.x can pass an int here; duck-type so both work.
        cur = self._seq_lens[layer_idx]
        def _len(x):
            return x.shape[0] if hasattr(x, "shape") else int(x)
        if cur == 0 and cache_position is not None:
            return _len(cache_position), 0
        if cache_position is not None:
            return cur + _len(cache_position), 0
        return cur, 0

    def is_initialized(self, layer_idx=0):
        return self._seq_lens[layer_idx] > 0

    def has_previous_state(self, layer_idx=0):
        # transformers 5.x qwen3_5_moe linear-attn path calls this. In hybrid
        # models, transformer-attn layers track progress via _seq_lens (set in
        # update()), DeltaNet layers track progress via conv/recurrent state
        # (set via update_conv_state / update_recurrent_state). Either signal
        # indicates the layer has been processed at least once.
        return (
            self._seq_lens[layer_idx] > 0
            or self.layers[layer_idx].conv_states is not None
            or self.layers[layer_idx].recurrent_states is not None
        )

    def update_conv_state(self, conv_state, layer_idx):
        """DeltaNet conv state setter. Called by Qwen3_5MoeGatedDeltaNet.forward
        (transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:473).

        Stores a reference: the caller's causal_conv1d_update mutates the same
        tensor in-place across decode steps, so persisting the reference is
        sufficient and avoids re-allocation per token.
        """
        self.layers[layer_idx].conv_states = conv_state

    def update_recurrent_state(self, recurrent_state, layer_idx):
        """DeltaNet recurrent state setter. Called after recurrent_gated_delta_rule
        (qwen3_5_moe modeling, line 535)."""
        self.layers[layer_idx].recurrent_states = recurrent_state

    @staticmethod
    def is_compileable():
        return False

    @property
    def max_batch_size(self):
        return 1

    @property
    def max_cache_len(self):
        return len(self.page_ids) * PAGE_SIZE

    def early_initialization(self, *args, **kwargs):
        pass

    def crop(self, max_length):
        any_change = False
        for i in range(len(self._seq_lens)):
            if self._seq_lens[i] > max_length:
                self._seq_lens[i] = max_length
                any_change = True
        if any_change:
            new_pages = self._pages_needed(max(self._seq_lens) if self._seq_lens else 0)
            while len(self.page_ids) > new_pages:
                self.pool.free_page(self.page_ids.pop())

    def batch_repeat_interleave(self, repeats):
        return self

    def batch_select_indices(self, indices):
        return self

    def reorder_cache(self, beam_idx):
        pass

    def offload(self):
        pass

    def prefetch(self, layer_idx):
        pass

    def reset(self):
        self.free()

    def free(self):
        for pid in self.page_ids:
            self.pool.free_page(pid)
        self.page_ids.clear()
        for i in range(len(self._seq_lens)):
            self._seq_lens[i] = 0
        for layer in self.layers:
            layer.conv_states = None
            layer.recurrent_states = None

    def __len__(self):
        return self.pool.num_layers

    def __iter__(self):
        for i in range(self.pool.num_layers):
            n = self._seq_lens[i]
            k = self.pool.read(self.page_ids, i, 0, n)
            v = self.pool.read(self.page_ids, i, 1, n)
            yield (k, v)

    def __getitem__(self, idx):
        n = self._seq_lens[idx]
        k = self.pool.read(self.page_ids, idx, 0, n)
        v = self.pool.read(self.page_ids, idx, 1, n)
        return (k, v)

    def __contains__(self, item):
        return False

    def __bool__(self):
        return any(s > 0 for s in self._seq_lens)
