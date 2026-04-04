# How to Write a Fused Dequant+Matmul Triton Kernel for K-Quants

*2026-04-04. Educational analysis. Nobody has published this yet — this would be novel work.*

---

## What "Fused" Means

A **non-fused** pipeline does dequant and matmul as separate steps:

```
GPU VRAM (Q4_K bytes) → Kernel 1: dequant to FP16 buffer → GPU VRAM (FP16) → Kernel 2: matmul → output
```

This reads expert weights from VRAM **twice**: once for dequant, once for matmul. At batch=1, the operation is memory-bandwidth-bound, so this doubles the wall time.

A **fused** kernel does both in one pass:

```
GPU VRAM (Q4_K bytes) → Single kernel: load tile, dequant in registers, matmul, accumulate → output
```

Weights are read from VRAM **once**. The dequant happens in registers/shared memory as data streams through. This is the ExLlamaV2 approach and the reason it's 1.6-1.85x faster than separate dequant+matmul.

---

## The Triton Matmul Structure

A standard Triton matmul kernel (from the [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)):

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,  # input/weight/output pointers
    M, N, K,               # matrix dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program instance computes a BLOCK_M x BLOCK_N tile of output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k in range(0, K, BLOCK_K):
        # Load tile from A: [BLOCK_M, BLOCK_K]
        a_tile = tl.load(a_ptr + offsets_a)  # activations (FP16)

        # Load tile from B: [BLOCK_K, BLOCK_N]
        b_tile = tl.load(b_ptr + offsets_b)  # weights (FP16)

        # Tile matmul: acc += a_tile @ b_tile
        acc += tl.dot(a_tile, b_tile)

    # Store result
    tl.store(c_ptr + offsets_c, acc.to(tl.float16))
```

---

## Where Dequant Gets Injected

In the fused version, instead of loading FP16 weights directly, you load Q4_K bytes and dequant them **inside the k-loop**:

```python
    for k in range(0, K, BLOCK_K):
        # Load activation tile (always FP16/BF16)
        a_tile = tl.load(a_ptr + offsets_a)

        # === FUSED DEQUANT: load Q4_K bytes, decode to FP16 in registers ===
        b_tile = dequant_q4k_tile(b_ptr, k, pid_n, BLOCK_K, BLOCK_N)

        # Same matmul as before
        acc += tl.dot(a_tile, b_tile)
```

The dequant function runs entirely in registers. No intermediate FP16 buffer in VRAM.

---

## The Hard Part: Q4_K Block Decoding in Triton

Q4_K structure (144 bytes per 256 elements):
```
d:         float16 (2 bytes)   — super-block scale
dmin:      float16 (2 bytes)   — super-block minimum
scales[12]: bytes              — 6-bit packed sub-block scales and mins
qs[128]:   bytes               — 4-bit packed values (2 per byte)
```

To dequant one value at position `idx` within a block:
1. Read `d` and `dmin` (2 bytes each)
2. Compute sub-block index: `sb = idx // 32`
3. Unpack 6-bit scale and min for this sub-block from `scales[12]`
4. Read 4-bit quant value from `qs[idx // 2]`, mask with `0x0F` or `>> 4`
5. Compute: `value = d * scale * q - dmin * min`

### Why This Is Hard in Triton

**Problem 1: Non-uniform memory access.** A Q4_K block is 144 bytes containing 256 elements. The elements are NOT at uniform byte offsets — you need the header (d, dmin, scales) plus variable-position nibbles. Triton's `tl.load` works best with uniform strides.

**Problem 2: Bit manipulation.** Extracting 4-bit nibbles requires shifts and masks. Triton supports `>>` and `&` on integers, but the 6-bit scale unpacking is particularly gnarly:

```python
# Scales packing (6-bit values for 8 sub-blocks):
# bytes 0-7: low 4 bits of scale[i] | (low 4 bits of min[i] << 4)
# bytes 8-11: high 2 bits packed as (s_hi | (m_hi << 2)) at bit positions

def unpack_scales_q4k(scales_bytes, sub_block_idx):
    """Unpack 6-bit scale and min for one sub-block."""
    # Low 4 bits
    s_lo = scales_bytes[sub_block_idx] & 0x0F
    m_lo = (scales_bytes[sub_block_idx] >> 4) & 0x0F
    # High 2 bits
    byte_idx = 8 + (sub_block_idx // 2)
    shift = (sub_block_idx % 2) * 4
    packed_hi = (scales_bytes[byte_idx] >> shift) & 0x0F
    s_hi = packed_hi & 0x03
    m_hi = (packed_hi >> 2) & 0x03
    # Combine
    scale = s_lo | (s_hi << 4)  # 6-bit value
    min_val = m_lo | (m_hi << 4)  # 6-bit value
    return scale, min_val
```

In Triton, this would be vectorized across the tile, but the irregular byte indexing requires careful pointer arithmetic.

**Problem 3: Tile alignment.** A BLOCK_K of 32 or 64 elements doesn't align cleanly with Q4_K blocks (256 elements each). You either:
- Use BLOCK_K=256 (matches Q4_K blocks but may be too large for shared memory)
- Handle partial blocks at tile boundaries (complex masking logic)
- Pre-pad to 256 alignment (wastes compute)

---

## Sketch of the Fused Kernel

```python
@triton.jit
def fused_q4k_matmul_kernel(
    x_ptr,        # [M, K] activations in BF16
    w_ptr,        # [N, n_blocks, 144] Q4_K packed weights (raw bytes)
    out_ptr,      # [M, N] output
    M, N, K,
    n_blocks,     # K // 256
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused Q4_K dequant + matmul. One VRAM read per weight."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Output tile accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in chunks of 256 (one Q4_K block per chunk)
    for block_idx in range(n_blocks):
        k_start = block_idx * 256

        # Load activation tile: [BLOCK_M, 256]
        x_tile = tl.load(x_ptr + pid_m * BLOCK_M * K + k_start, ...)

        # Load Q4_K block headers for BLOCK_N output rows
        # Each row has its own Q4_K block at this k position
        # d[BLOCK_N], dmin[BLOCK_N], scales[BLOCK_N, 12], qs[BLOCK_N, 128]
        row_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        block_byte_offset = row_offsets * n_blocks * 144 + block_idx * 144

        d = tl.load(w_ptr + block_byte_offset).to(tl.float16)       # [BLOCK_N]
        dmin = tl.load(w_ptr + block_byte_offset + 2).to(tl.float16) # [BLOCK_N]

        # For each of 8 sub-blocks of 32 elements:
        for sb in range(8):
            # Unpack 6-bit scale and min for this sub-block
            # ... (complex bit manipulation, vectorized across BLOCK_N rows)

            # Load 32 nibble values from qs
            # ... (load 16 bytes, unpack low/high nibbles)

            # Dequant: w_tile[:, sb*32:(sb+1)*32] = d * scale * q - dmin * min
            w_sub = ...  # [BLOCK_N, 32] in BF16

            # Partial dot product for this sub-block
            x_sub = x_tile[:, sb*32:(sb+1)*32]  # [BLOCK_M, 32]
            acc += tl.dot(x_sub, tl.trans(w_sub))

    tl.store(out_ptr + ..., acc.to(tl.bfloat16))
```

---

## Effort Estimate

| Component | Difficulty | LOC | Notes |
|-----------|-----------|-----|-------|
| Q4_K block header loading | Easy | ~20 | Structured byte loads |
| 4-bit nibble unpacking | Medium | ~15 | Shift/mask, vectorized |
| 6-bit scale unpacking | Hard | ~30 | Irregular byte indexing, cross-byte reads |
| Tile loop structure | Medium | ~40 | Based on Triton matmul tutorial |
| Sub-block iteration | Medium | ~20 | 8 sub-blocks of 32 within the 256-block |
| Autotuning configs | Easy | ~10 | BLOCK_M, BLOCK_N grid search |
| **Q4_K total** | | **~135** | |
| Q5_K variant | Medium | +40 | Same structure + 5th bit from qh bytes |
| Q6_K variant | Medium | +50 | Different layout (ql + qh + int8 scales) |
| Q8_0 variant | Easy | +20 | Simplest: just scale * int8 |
| **All K-quants** | | **~245** | |
| Tests + validation | Medium | ~100 | Accuracy vs reference, perf benchmarks |
| **Total** | | **~350 LOC** | |

### Calendar time: 2-4 days for a skilled Triton developer

The hard part isn't the LOC — it's debugging the bit manipulation. One wrong shift and the output is garbage. You'd need:
1. A Python reference implementation (we already have this in `gguf_quant.py:parse_q4k_block`)
2. Element-by-element validation against the reference
3. Performance tuning: BLOCK_M/BLOCK_N selection, shared memory usage

---

## Why It Would Be Valuable

1. **Novel contribution** — nobody has published this
2. **Pure Python/Triton** — no C++ build complexity, works on any GPU with Triton
3. **Composable** — Triton kernels can be fused with activations, routing, etc.
4. **Portable** — Triton targets NVIDIA, AMD, and Intel GPUs
5. **Maintainable** — ~350 LOC Python vs thousands of LOC CUDA in llama.cpp

### But the practical calculus:

At batch=1 decode, fused Triton would perform identically to llama.cpp CUDA (both bandwidth-bound). The advantage is maintainability and portability, not speed. For a production system that needs to ship now, wrapping battle-tested llama.cpp kernels is lower risk. The Triton kernel is a great learning project and potential OSS contribution.

---

## Building Blocks Available Today

| Piece | Source | License |
|-------|--------|---------|
| Q4_K bit unpacking in Triton | [ComfyUI-GGUF PR #336](https://github.com/city96/ComfyUI-GGUF/pull/336) | Apache-2.0 |
| Fused dequant+matmul template | [GPTQ-Triton](https://github.com/fpgaminer/GPTQ-triton) | Apache-2.0 |
| SplitK for small-M regime | [W4A16 paper (arxiv 2402.00025)](https://arxiv.org/abs/2402.00025) | Academic |
| Reference dequant (Python) | tinyserve `gguf_quant.py:parse_q4k_block` | MIT (ours) |
| Fused HQQ dequant+GEMM | [PyTorch AO PR #153](https://github.com/pytorch/ao/pull/153) | BSD-3 |

The recipe: take GPTQ-Triton's matmul structure, replace the GPTQ dequant with ComfyUI-GGUF's K-quant dequant, validate against our `parse_q4k_block` reference.
