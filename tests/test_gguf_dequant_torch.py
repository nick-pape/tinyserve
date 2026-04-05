"""Tests for pure-PyTorch GGUF dequantization fallback."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

# Q4_K block layout constants
Q4K_BLOCK_BYTES = 144
Q4K_BLOCK_ELEMENTS = 256


def _build_q4k_block_gguf(values_f32: np.ndarray) -> bytes:
    """Build a Q4_K block in the actual llama.cpp/GGUF wire format.

    This uses the real get_scale_min_k4 layout used by llama.cpp and city96:
      bytes 0-3:   sc[0..3] in low 6 bits, upper 2 bits of sc[4..7] in bits 6-7
      bytes 4-7:   min[0..3] in low 6 bits, upper 2 bits of min[4..7] in bits 6-7
      bytes 8-11:  sc[i+4] bits 0-3 in low nibble, min[i+4] bits 0-3 in high nibble
      bytes 16-143: 4-bit quantized values (2 per byte)
    """
    assert values_f32.shape == (256,)

    sub_blocks = values_f32.reshape(8, 32)
    sub_mins = sub_blocks.min(axis=1)
    sub_maxs = sub_blocks.max(axis=1)
    sub_ranges = sub_maxs - sub_mins

    max_range = max(float(sub_ranges.max()), 1e-10)
    min_min = float(sub_mins.min())

    d = np.float16(max_range / 63.0)
    dmin = np.float16(-min_min / 63.0) if min_min < 0 else np.float16(0.0)
    d_f32 = float(d)
    dmin_f32 = float(dmin)

    raw_sc = np.zeros(8, dtype=np.uint8)
    raw_mn = np.zeros(8, dtype=np.uint8)
    for i in range(8):
        if d_f32 > 0:
            raw_sc[i] = min(63, max(0, int(round(sub_ranges[i] / d_f32))))
        if dmin_f32 > 0:
            raw_mn[i] = min(63, max(0, int(round(-sub_mins[i] / dmin_f32))))

    # Pack 8 sc + 8 mn (6 bits each) into 12 bytes using the real llama.cpp format:
    #   bytes 0-3: low 6 bits of sc[0..3], bits 6-7 = upper 2 bits of sc[4..7]
    #   bytes 4-7: low 6 bits of mn[0..3], bits 6-7 = upper 2 bits of mn[4..7]
    #   bytes 8-11: (sc[i+4] & 0x0F) | ((mn[i+4] & 0x0F) << 4)
    scales_bytes = bytearray(12)
    for i in range(4):
        scales_bytes[i] = (raw_sc[i] & 0x3F) | ((raw_sc[i + 4] >> 4) << 6)
        scales_bytes[4 + i] = (raw_mn[i] & 0x3F) | ((raw_mn[i + 4] >> 4) << 6)
        scales_bytes[8 + i] = (raw_sc[i + 4] & 0x0F) | ((raw_mn[i + 4] & 0x0F) << 4)

    # Quantize values to 4-bit using the real llama.cpp interleaved storage:
    # qs bytes are split into 4 chunks of 32 bytes; each chunk covers 64 elements:
    #   bytes chunk*32 + l (l=0..31): low nibble = element chunk*64+l,
    #                                 high nibble = element chunk*64+32+l
    qs = bytearray(128)
    for chunk in range(4):
        for l in range(32):
            sub0 = chunk * 2       # even sub-block for this chunk
            sub1 = chunk * 2 + 1   # odd sub-block for this chunk

            elem0 = chunk * 64 + l
            elem1 = chunk * 64 + 32 + l
            byte_idx = chunk * 32 + l

            sc0 = d_f32 * raw_sc[sub0]
            mn0 = dmin_f32 * raw_mn[sub0]
            q0 = int(round((values_f32[elem0] + mn0) / sc0)) if sc0 > 0 else 0
            q0 = max(0, min(15, q0))

            sc1 = d_f32 * raw_sc[sub1]
            mn1 = dmin_f32 * raw_mn[sub1]
            q1 = int(round((values_f32[elem1] + mn1) / sc1)) if sc1 > 0 else 0
            q1 = max(0, min(15, q1))

            qs[byte_idx] = (q0 & 0x0F) | ((q1 & 0x0F) << 4)

    block = struct.pack("<e", d) + struct.pack("<e", dmin) + bytes(scales_bytes) + bytes(qs)
    assert len(block) == Q4K_BLOCK_BYTES
    return block


def _dequant_q4k_reference(block_bytes: bytes) -> np.ndarray:
    """Reference Q4_K dequant using the real GGUF format (mirrors city96 logic in Python)."""
    d = struct.unpack_from("<e", block_bytes, 0)[0]
    dmin = struct.unpack_from("<e", block_bytes, 2)[0]
    scales_raw = block_bytes[4:16]
    qs_raw = block_bytes[16:144]

    sc = np.zeros(8, dtype=np.float32)
    mn = np.zeros(8, dtype=np.float32)
    for j in range(4):
        sc[j] = d * (scales_raw[j] & 0x3F)
        mn[j] = dmin * (scales_raw[4 + j] & 0x3F)
        sc_hi = (scales_raw[8 + j] & 0x0F) | ((scales_raw[j] >> 6) << 4)
        mn_hi = (scales_raw[8 + j] >> 4) | ((scales_raw[4 + j] >> 6) << 4)
        sc[j + 4] = d * sc_hi
        mn[j + 4] = dmin * mn_hi

    # Real llama.cpp interleaved qs storage: each 32-byte chunk covers 64 elements
    #   low nibble of byte chunk*32+l  -> element chunk*64+l   (sub-block 2*chunk)
    #   high nibble of byte chunk*32+l -> element chunk*64+32+l (sub-block 2*chunk+1)
    values = np.empty(256, dtype=np.float32)
    for chunk in range(4):
        for l in range(32):
            byte_val = qs_raw[chunk * 32 + l]
            sub0 = chunk * 2
            sub1 = chunk * 2 + 1
            values[chunk * 64 + l] = sc[sub0] * (byte_val & 0x0F) - mn[sub0]
            values[chunk * 64 + 32 + l] = sc[sub1] * ((byte_val >> 4) & 0x0F) - mn[sub1]
    return values


class TestDequantQ8_0KnownValues:
    """Q8_0 block: 2 bytes f16 scale + 32 bytes int8 quants = 34 bytes per 32 elements."""

    def test_dequant_q8_0_known_values(self):
        """Manually constructed Q8_0 block dequants to exact expected values."""
        from tinyserve.gguf_dequant import dequant_tensor

        GGML_Q8_0 = 8
        scale_f16 = np.float16(2.0)
        quants = np.arange(1, 33, dtype=np.int8)

        block = struct.pack("<e", scale_f16) + quants.tobytes()
        assert len(block) == 34

        result = dequant_tensor(block, GGML_Q8_0, (32,))

        assert result.shape == (32,)
        assert result.dtype == torch.float32

        expected = torch.tensor(quants.astype(np.float32) * float(scale_f16))
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=0.0)

    def test_dequant_q8_0_accepts_uint8_tensor(self):
        """dequant_tensor accepts torch.Tensor of dtype uint8 as data input."""
        from tinyserve.gguf_dequant import dequant_tensor

        GGML_Q8_0 = 8
        scale_f16 = np.float16(1.0)
        quants = np.zeros(32, dtype=np.int8)

        block_bytes = struct.pack("<e", scale_f16) + quants.tobytes()
        block_tensor = torch.frombuffer(bytearray(block_bytes), dtype=torch.uint8)

        result_bytes = dequant_tensor(block_bytes, GGML_Q8_0, (32,))
        result_tensor = dequant_tensor(block_tensor, GGML_Q8_0, (32,))

        torch.testing.assert_close(result_bytes, result_tensor)

    def test_dequant_q8_0_negative_quants(self):
        """Q8_0 with negative int8 quants produces correct negative float output."""
        from tinyserve.gguf_dequant import dequant_tensor

        GGML_Q8_0 = 8
        scale_f16 = np.float16(0.5)
        quants = np.array([-16, -8, 0, 8, 16] + [0] * 27, dtype=np.int8)

        block = struct.pack("<e", scale_f16) + quants.tobytes()
        result = dequant_tensor(block, GGML_Q8_0, (32,))

        expected = torch.tensor(quants.astype(np.float32) * float(scale_f16))
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=0.0)


class TestDequantQ4KMatchesReference:
    """Q4_K dequant must match a Python reference using the same GGUF wire format."""

    def test_dequant_q4k_matches_reference_single_block(self):
        """Single Q4_K block built in real GGUF format matches Python reference decode."""
        from tinyserve.gguf_dequant import dequant_tensor

        GGML_Q4_K = 12
        np.random.seed(7)
        values = np.random.uniform(-2.0, 2.0, 256).astype(np.float32)
        block = _build_q4k_block_gguf(values)
        assert len(block) == Q4K_BLOCK_BYTES

        result = dequant_tensor(block, GGML_Q4_K, (256,))
        reference = torch.from_numpy(_dequant_q4k_reference(block))

        assert result.shape == (256,)
        assert result.dtype == torch.float32
        torch.testing.assert_close(result, reference, atol=1e-4, rtol=1e-4)

    def test_dequant_q4k_matches_reference_multi_block(self):
        """Multiple Q4_K blocks in GGUF format match Python reference decode."""
        from tinyserve.gguf_dequant import dequant_tensor

        GGML_Q4_K = 12
        np.random.seed(99)
        n_blocks = 4
        all_blocks = b""
        for _ in range(n_blocks):
            vals = np.random.uniform(-3.0, 3.0, 256).astype(np.float32)
            all_blocks += _build_q4k_block_gguf(vals)

        total_elements = n_blocks * 256
        shape = (total_elements,)

        result = dequant_tensor(all_blocks, GGML_Q4_K, shape)
        reference = np.concatenate([
            _dequant_q4k_reference(all_blocks[b * Q4K_BLOCK_BYTES:(b + 1) * Q4K_BLOCK_BYTES])
            for b in range(n_blocks)
        ])

        assert result.shape == shape
        torch.testing.assert_close(result, torch.from_numpy(reference), atol=1e-4, rtol=1e-4)

    def test_dequant_q4k_2d_shape(self):
        """Q4_K dequant correctly reshapes to 2D output."""
        from tinyserve.gguf_dequant import dequant_tensor

        GGML_Q4_K = 12
        np.random.seed(11)
        n_blocks = 2
        all_blocks = b""
        for _ in range(n_blocks):
            vals = np.random.uniform(-1.0, 1.0, 256).astype(np.float32)
            all_blocks += _build_q4k_block_gguf(vals)

        result = dequant_tensor(all_blocks, GGML_Q4_K, (2, 256))
        assert result.shape == (2, 256)
        assert result.dtype == torch.float32

    def test_dequant_q4k_round_trip_correlation(self):
        """Q4_K dequant output correlates strongly with the original float values."""
        from tinyserve.gguf_dequant import dequant_tensor

        GGML_Q4_K = 12
        np.random.seed(42)
        values = np.random.uniform(-2.0, 2.0, 256).astype(np.float32)
        block = _build_q4k_block_gguf(values)

        result = dequant_tensor(block, GGML_Q4_K, (256,)).numpy()
        corr = np.corrcoef(values, result)[0, 1]
        # Q4_K with simplified 8-sub-block encoding achieves ~0.85-0.90 correlation;
        # a lower bound of 0.80 confirms dequant is faithful without demanding real-encoder precision.
        assert corr > 0.80, f"Correlation {corr:.4f} too low — dequant is not faithful"


class TestDequantUnsupportedRaises:
    def test_unsupported_type_raises_value_error(self):
        """Unsupported GGML type raises ValueError with type id in message."""
        from tinyserve.gguf_dequant import dequant_tensor

        with pytest.raises(ValueError, match="99"):
            dequant_tensor(b"\x00" * 100, 99, (32,))

    def test_type_0_not_supported(self):
        """F32 (type 0) raises ValueError — caller should cast directly."""
        from tinyserve.gguf_dequant import dequant_tensor

        with pytest.raises(ValueError):
            dequant_tensor(b"\x00" * 4, 0, (1,))


class TestDequantLegacyTypes:
    """Smoke tests for Q4_0, Q4_1, Q5_0, Q5_1 — verify shape and no NaN/Inf."""

    def _make_q4_0_blocks(self, n: int) -> bytes:
        """n Q4_0 blocks: 2 bytes f16 scale + 16 bytes packed 4-bit = 18 bytes/32 elements."""
        blocks = b""
        for i in range(n):
            scale = struct.pack("<e", np.float16(0.5 + i * 0.1))
            qs = bytes([0xAB] * 16)
            blocks += scale + qs
        return blocks

    def _make_q4_1_blocks(self, n: int) -> bytes:
        """n Q4_1 blocks: 2 f16 (d+m) + 16 bytes packed 4-bit = 20 bytes/32 elements."""
        blocks = b""
        for i in range(n):
            d = struct.pack("<e", np.float16(0.5))
            m = struct.pack("<e", np.float16(0.1))
            qs = bytes([0x12] * 16)
            blocks += d + m + qs
        return blocks

    def _make_q5_0_blocks(self, n: int) -> bytes:
        """n Q5_0 blocks: 2 f16 scale + 4 bytes qh + 16 bytes ql = 22 bytes/32 elements."""
        blocks = b""
        for i in range(n):
            d = struct.pack("<e", np.float16(0.5))
            qh = bytes([0x00] * 4)
            qs = bytes([0xAB] * 16)
            blocks += d + qh + qs
        return blocks

    def _make_q5_1_blocks(self, n: int) -> bytes:
        """n Q5_1 blocks: 2 f16 d + 2 f16 m + 4 bytes qh + 16 bytes ql = 24 bytes/32 elements."""
        blocks = b""
        for i in range(n):
            d = struct.pack("<e", np.float16(0.5))
            m = struct.pack("<e", np.float16(0.1))
            qh = bytes([0x00] * 4)
            qs = bytes([0xAB] * 16)
            blocks += d + m + qh + qs
        return blocks

    def test_q4_0_shape_no_nan(self):
        from tinyserve.gguf_dequant import dequant_tensor

        data = self._make_q4_0_blocks(4)
        result = dequant_tensor(data, 2, (128,))
        assert result.shape == (128,)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_q4_1_shape_no_nan(self):
        from tinyserve.gguf_dequant import dequant_tensor

        data = self._make_q4_1_blocks(4)
        result = dequant_tensor(data, 3, (128,))
        assert result.shape == (128,)
        assert not torch.isnan(result).any()

    def test_q5_0_shape_no_nan(self):
        from tinyserve.gguf_dequant import dequant_tensor

        data = self._make_q5_0_blocks(4)
        result = dequant_tensor(data, 6, (128,))
        assert result.shape == (128,)
        assert not torch.isnan(result).any()

    def test_q5_1_shape_no_nan(self):
        from tinyserve.gguf_dequant import dequant_tensor

        data = self._make_q5_1_blocks(4)
        result = dequant_tensor(data, 7, (128,))
        assert result.shape == (128,)
        assert not torch.isnan(result).any()
