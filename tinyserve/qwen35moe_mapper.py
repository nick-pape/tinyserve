"""Qwen3.5MoE GGUF → HuggingFace weight mapping.

Derived from llama.cpp PR #19468 (convert_hf_to_gguf.py, Qwen3_5MoeTextModel).
This module inverts those transforms to load GGUF weights into the HF model.

Key transforms to invert:
- Norm weights: convert_hf_to_gguf.py does `data_torch = data_torch + 1` for every
  *.norm.weight EXCEPT linear_attn.norm.weight, so GGUF stores `w_hf + 1`. To recover
  the HF weight we SUBTRACT 1. (HF's Qwen3_5MoeRMSNorm.forward then computes
  `(1.0 + self.weight) * normalize(x)` so the effective scale stays `w_hf + 1`.)
- V-head reorder: GGUF uses tiled order, HF uses grouped → reorder back
- A_log: GGUF has -exp(A_log), HF expects A_log → take -log(-x)
- QKV: GGUF has fused attn_qkv, HF expects separate or fused in_proj_qkv
- Weight transpose: GGUF stores (out, in), HF nn.Linear expects (out, in) — same! No transpose.
  (ggml's ne[0] = cols = in_features, but the raw byte layout IS row-major matching PyTorch)
"""

from __future__ import annotations

import logging
import re

import torch

logger = logging.getLogger(__name__)

# GGUF canonical name → HF parameter name
# {bid} is replaced with layer index
_GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

_LAYER_MAP = {
    # Norms
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",

    # Full attention (every full_attention_interval-th layer)
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",

    # Linear attention (Gated DeltaNet) — the recurrent layers
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",

    # MoE router
    "ffn_gate_inp.weight": "mlp.gate.weight",

    # Shared expert
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
}

# Tensor names that need +1 offset inverted (norm weights stored as w-1 in GGUF)
_NORM_OFFSET_NAMES = {
    "attn_norm.weight",
    "post_attention_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    # NOTE: ssm_norm.weight does NOT get the +1 offset
}

# Fused expert tensors — handled by MmapExpertStore, not this mapper
_FUSED_EXPERT_NAMES = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}

# Linear attention tensors that need V-head reorder inversion.
# These are stored in GGUF with tiled V-head order; HF expects grouped order.
# Quantized (Q4_K) rows cannot be permuted in-place, so these must be
# dequantized to BF16 before applying the inverse reorder.
#
# "full" = all rows/cols are V-head ordered
# "v_portion" = only the V slice of a fused QKV is reordered
# "out_proj" = columns (dim=1) are reordered instead of rows
# "a_log" = special transform: GGUF stores -exp(A_log), HF expects A_log
_VHEAD_REORDER_MODE: dict[str, str] = {
    "attn_qkv.weight": "v_portion",    # linear_attn.in_proj_qkv — only V rows
    "attn_gate.weight": "full",         # linear_attn.in_proj_z
    "ssm_alpha.weight": "full",         # linear_attn.in_proj_a
    "ssm_beta.weight": "full",          # linear_attn.in_proj_b
    "ssm_out.weight": "out_proj",       # linear_attn.out_proj — cols reordered
    "ssm_conv1d.weight": "v_portion",   # linear_attn.conv1d — V portion
    "ssm_a": "a_log",                   # linear_attn.A_log — special transform
    "ssm_dt.bias": "full",              # linear_attn.dt_bias
}


def inverse_vhead_reorder_bytes(
    raw: bytes | torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    num_rows: int,
    bytes_per_row: int,
) -> bytes:
    """Reorder raw quantized bytes by permuting rows (V-head tiled → grouped).

    Each row is bytes_per_row contiguous bytes. The permutation operates
    on row indices without touching block internals.
    """
    if isinstance(raw, torch.Tensor):
        t = raw.reshape(num_rows, bytes_per_row)
    else:
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(num_rows, bytes_per_row)

    r = num_v_heads // num_k_heads
    head_dim_rows = num_rows // num_v_heads  # rows per V-head

    # [num_rows, bpr] -> [r, nk, head_dim_rows, bpr]
    t = t.reshape(r, num_k_heads, head_dim_rows, bytes_per_row)
    # tiled [r, nk] -> grouped [nk, r]
    t = t.transpose(0, 1).contiguous()
    # [nk, r, head_dim_rows, bpr] -> [num_rows, bpr]
    t = t.reshape(num_rows, bytes_per_row)
    return t.flatten().numpy().tobytes()


def inverse_vhead_reorder(
    tensor: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    dim: int = 0,
) -> torch.Tensor:
    """Convert V-head dimension from tiled (GGUF) back to grouped (HF) order.

    llama.cpp stores V-heads in tiled order:
      tiled:   [G0_v0, G1_v0, ..., Gn_v0, G0_v1, G1_v1, ..., Gn_v1]
    HF expects grouped order:
      grouped: [G0_v0, G0_v1, G1_v0, G1_v1, ...]

    Args:
        tensor: weight tensor to reorder
        num_k_heads: number of K heads (= number of groups)
        num_v_heads: total number of V heads across all groups
        dim: dimension along which V-heads are laid out (0 for row weights, 1 for out_proj)
    """
    r = num_v_heads // num_k_heads
    head_dim = tensor.shape[dim] // num_v_heads

    # Split the V-head dimension: [num_v_heads * head_dim] -> [r, num_k_heads, head_dim, ...]
    shape = list(tensor.shape)
    shape[dim:dim + 1] = [r, num_k_heads, head_dim]
    t = tensor.reshape(shape)

    # Swap r and num_k_heads axes: tiled [r, nk] -> grouped [nk, r]
    t = t.transpose(dim, dim + 1).contiguous()

    # Flatten back to original shape
    t = t.reshape(tensor.shape)
    return t


def apply_vhead_transform(
    tensor: torch.Tensor,
    mode: str,
    num_k_heads: int,
    num_v_heads: int,
    num_q_heads: int,
    k_head_dim: int,
    v_head_dim: int,
    q_head_dim: int,
) -> torch.Tensor:
    """Apply the appropriate V-head inverse transform for a given tensor mode.

    Args:
        tensor: dequantized BF16 tensor
        mode: one of "full", "v_portion", "out_proj", "a_log"
        num_k_heads: linear_num_key_heads from HF config
        num_v_heads: linear_num_value_heads from HF config
        num_q_heads: number of Q heads for in_proj_qkv (= num_attention_heads)
        k_head_dim: linear_key_head_dim from HF config
        v_head_dim: linear_value_head_dim from HF config
        q_head_dim: head_dim for Q heads (= hidden_size // num_attention_heads)
    """
    if mode == "a_log":
        # GGUF stores -exp(A_log), HF expects A_log = log(-gguf_value)
        return torch.log(torch.abs(tensor) + 1e-10)

    if mode == "full":
        return inverse_vhead_reorder(tensor, num_k_heads, num_v_heads, dim=0)

    if mode == "out_proj":
        return inverse_vhead_reorder(tensor, num_k_heads, num_v_heads, dim=1)

    if mode == "v_portion":
        # Fused [Q | K | V] channels. In Qwen3.5MoE's GatedDeltaNet, Q heads use
        # the *linear-attn key* head_dim (k_head_dim), NOT the main-attention
        # head_dim — that's a property of the linear-attention block, not the
        # global model config. For Qwen3.6 (Pro 4500 hardware, 2026-05-11)
        # main attn head_dim is 256 but linear-attn key head_dim is 128, so
        # the old formula `q_dim = num_q_heads * q_head_dim` over-counted Q by
        # 4096 and the channel dim never matched any tensor axis.
        qk_dim_each = num_k_heads * k_head_dim  # Q == K dim in GatedDeltaNet
        v_dim = num_v_heads * v_head_dim
        total_channels = 2 * qk_dim_each + v_dim
        non_v_dim = 2 * qk_dim_each

        # The channel axis isn't always dim 0: attn_qkv.weight stores channels
        # on dim 0, but ssm_conv1d.weight in GGUF stores them on a later axis
        # (the kernel_size dim ends up first). Locate the channel axis by
        # matching the expected total, then move it to the front for the
        # slice + reorder, then move it back.
        channel_axis = next(
            (i for i, s in enumerate(tensor.shape) if s == total_channels),
            None,
        )
        if channel_axis is None:
            logger.warning(
                "V-portion size mismatch: expected channel axis of size %d (=2*%d K-shaped + %d V) "
                "in tensor of shape %s. Skipping V-head reorder.",
                total_channels, qk_dim_each, v_dim, list(tensor.shape),
            )
            return tensor

        if channel_axis != 0:
            # Bring channels to dim 0 for slicing. We DON'T transpose back at
            # the end: the downstream gguf_loader reshape to the HF param
            # shape (e.g. conv1d (channels, 1, kernel_size)) only produces
            # correct element layout when channels are already at dim 0.
            # For the QKV weight (channel_axis=0) this branch is skipped so
            # the original (channels, hidden) layout is preserved.
            tensor = tensor.transpose(0, channel_axis).contiguous()

        non_v_part = tensor[:non_v_dim]
        v_part = tensor[non_v_dim:]
        v_part = inverse_vhead_reorder(v_part, num_k_heads, num_v_heads, dim=0)
        return torch.cat([non_v_part, v_part], dim=0)

    raise ValueError(f"Unknown vhead reorder mode: {mode!r}")


def map_gguf_to_hf(gguf_name: str) -> tuple[str | None, bool, bool, str | None]:
    """Map a GGUF tensor name to its HuggingFace parameter path.

    Returns:
        (hf_name, needs_norm_offset, is_fused_expert, vhead_reorder_mode)

        vhead_reorder_mode is one of None, "full", "v_portion", "out_proj", "a_log".
        hf_name is None if the tensor is unmapped.
    """
    # Global tensors
    if gguf_name in _GLOBAL_MAP:
        is_norm = gguf_name.endswith("norm.weight")
        return _GLOBAL_MAP[gguf_name], is_norm, False, None

    # Layer tensors: blk.{bid}.{suffix}
    m = re.match(r"blk\.(\d+)\.(.+)", gguf_name)
    if not m:
        return None, False, False, None

    bid = int(m.group(1))
    suffix = m.group(2)

    # Fused expert tensors
    if suffix in _FUSED_EXPERT_NAMES:
        return None, False, True, None

    hf_suffix = _LAYER_MAP.get(suffix)
    if hf_suffix is None:
        return None, False, False, None

    hf_name = f"model.layers.{bid}.{hf_suffix}"
    needs_offset = suffix in _NORM_OFFSET_NAMES
    vhead_mode = _VHEAD_REORDER_MODE.get(suffix)
    return hf_name, needs_offset, False, vhead_mode
