"""Tests for MmapExpertStore — zero-copy GGUF expert storage."""

import struct
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import requires_cuda
from tests.test_gguf_reader import _create_synthetic_gguf
from tinyserve.gguf_reader import GGML_TYPES


def _create_fused_gguf(path, n_layers=2, n_experts=4, ggml_type=0):
    """Create a minimal GGUF v3 file with fused expert tensors.

    Uses F32 (ggml_type=0) by default.
    Each fused tensor has shape (out_dim, in_dim, n_experts) = (32, 64, n_experts).
    Tensor data is float32 values: proj_idx * 0.1 + expert_idx * 0.01 (broadcast).
    """
    type_name, bytes_per_block, block_size = GGML_TYPES.get(ggml_type, ("F32", 4, 1))

    out_dim, in_dim = 32, 64
    shape_3d = (out_dim, in_dim, n_experts)
    n_elements = out_dim * in_dim * n_experts
    n_blocks = (n_elements + block_size - 1) // block_size
    tensor_nbytes = n_blocks * bytes_per_block

    projections = ["gate", "up", "down"]
    n_tensors = n_layers * len(projections)
    n_kv = 2

    tensor_infos = []
    data_offset = 0
    for layer in range(n_layers):
        for proj in projections:
            name = f"blk.{layer}.ffn_{proj}_exps.weight"
            tensor_infos.append((name, shape_3d, ggml_type, data_offset))
            data_offset += tensor_nbytes

    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x46554747))
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        # Metadata
        encoded = "test-fused-model".encode("utf-8")
        f.write(struct.pack("<Q", len("general.name".encode("utf-8"))))
        f.write("general.name".encode("utf-8"))
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

        f.write(struct.pack("<Q", len("llama.expert_count".encode("utf-8"))))
        f.write("llama.expert_count".encode("utf-8"))
        f.write(struct.pack("<I", 4))
        f.write(struct.pack("<I", n_experts))

        # Tensor infos
        for name, shape, gtype, offset in tensor_infos:
            encoded_name = name.encode("utf-8")
            f.write(struct.pack("<Q", len(encoded_name)))
            f.write(encoded_name)
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", gtype))
            f.write(struct.pack("<Q", offset))

        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        # Tensor data: F32 values, distinct per projection
        import numpy as np
        for layer in range(n_layers):
            for proj_idx, proj in enumerate(projections):
                # Shape (out_dim, in_dim, n_experts): each expert column gets value
                # layer*1.0 + proj_idx*0.1 + expert_idx*0.01
                data = np.zeros((out_dim, in_dim, n_experts), dtype=np.float32)
                for e in range(n_experts):
                    data[:, :, e] = layer * 1.0 + proj_idx * 0.1 + e * 0.01
                f.write(data.tobytes())

    return tensor_infos


class TestMmapStoreFromPerExpertGGUF:
    def test_num_layers_num_experts_from_synthetic_gguf(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=2, n_experts=4)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert store.num_layers == 2
        assert store.num_experts == 4
        store.close()

    def test_expert_bytes_positive(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert store.expert_bytes > 0
        store.close()

    def test_expert_bytes_equals_sum_of_projections(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        # F32, shape (64, 128) => 64*128*4 bytes per projection, 3 projections
        expected = 64 * 128 * 4 * 3
        assert store.expert_bytes == expected
        store.close()

    def test_buffer_expert_bytes_equals_expert_bytes(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert store.buffer_expert_bytes == store.expert_bytes
        store.close()


class TestMmapStoreInterfaceAttributes:
    def test_has_layout(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.expert_store import TensorLayout
        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert isinstance(store.layout, TensorLayout)
        store.close()

    def test_layout_has_gate_up_down_specs(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert set(store.layout.specs.keys()) == {"gate", "up", "down"}
        store.close()

    def test_layout_specs_are_uint8(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        for name, (shape, dtype) in store.layout.specs.items():
            assert dtype == torch.uint8, f"{name} spec dtype should be uint8"
        store.close()

    def test_has_bf16_layout(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.expert_store import TensorLayout
        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert isinstance(store._bf16_layout, TensorLayout)
        store.close()

    def test_bf16_layout_same_as_layout(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert store._bf16_layout is store.layout
        store.close()

    def test_has_ggml_types(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert isinstance(store.ggml_types, dict)
        assert set(store.ggml_types.keys()) == {"gate", "up", "down"}
        store.close()

    def test_has_proj_shapes(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert isinstance(store.proj_shapes, dict)
        assert set(store.proj_shapes.keys()) == {"gate", "up", "down"}
        for name, shape in store.proj_shapes.items():
            assert len(shape) == 2
        store.close()

    def test_fp8_property_returns_false(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert store._fp8 is False
        store.close()

    def test_allocate_buffer_returns_expert_buffer(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.expert_store import ExpertBuffer
        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        buf = store.allocate_buffer(torch.device("cpu"))
        assert isinstance(buf, ExpertBuffer)
        store.close()


class TestMmapStoreGetExpertData:
    def test_get_expert_data_returns_tensor(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=2, n_experts=4)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        data = store.get_expert_data(0, 0)
        assert isinstance(data, torch.Tensor)
        store.close()

    def test_get_expert_data_correct_size(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        data = store.get_expert_data(0, 0)
        assert data.numel() == store.expert_bytes
        store.close()

    def test_get_expert_data_is_uint8(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        data = store.get_expert_data(0, 0)
        assert data.dtype == torch.uint8
        store.close()

    def test_get_expert_data_different_experts_differ(self, tmp_path):
        """Different (layer, expert) pairs should produce different raw bytes."""
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=2, n_experts=4)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        d00 = store.get_expert_data(0, 0)
        d01 = store.get_expert_data(0, 1)
        assert not torch.equal(d00, d01)
        store.close()

    def test_get_expert_data_matches_gguf_pattern(self, tmp_path):
        """Layer=0, expert=0, gate projection should match byte pattern from _create_synthetic_gguf."""
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2, ggml_type=0)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        data = store.get_expert_data(0, 0)
        # F32 shape (64,128) = 32768 bytes per projection
        # gate projection (proj_idx=0): pattern_byte = (0*100 + 0*10 + 0) & 0xFF = 0
        gate_nbytes = store.layout.sizes["gate"]
        gate_bytes = data[:gate_nbytes]
        assert gate_bytes.tolist() == [0] * gate_nbytes
        store.close()


@requires_cuda
class TestMmapStoreCopyToBuffer:
    def test_copy_to_buffer_fills_gpu_buffer(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        device = torch.device("cuda")
        buf = store.allocate_buffer(device)
        store.copy_to_buffer(buf, 0, 1)  # expert 1 has non-zero pattern bytes
        assert buf.packed.device.type == "cuda"
        assert buf.packed.numel() == store.expert_bytes
        store.close()

    def test_copy_to_buffer_non_zero_data(self, tmp_path):
        """Expert 1 of layer 0 has pattern byte 10, so packed should be non-zero."""
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        device = torch.device("cuda")
        buf = store.allocate_buffer(device)
        store.copy_to_buffer(buf, 0, 1)
        assert buf.packed.sum().item() > 0
        store.close()

    def test_copy_to_buffer_non_blocking(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        device = torch.device("cuda")
        buf = store.allocate_buffer(device)
        store.copy_to_buffer(buf, 0, 1, non_blocking=True)
        torch.cuda.synchronize()
        assert buf.packed.sum().item() > 0
        store.close()

    def test_copy_to_buffer_slot(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.expert_cache import ExpertCache
        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        device = torch.device("cuda")
        cache = ExpertCache(
            capacity=4,
            expert_bytes=store.expert_bytes,
            device=device,
        )
        store.copy_to_buffer_slot(cache, 0, 0, 1)
        torch.cuda.synchronize()
        assert cache._packed[0].sum().item() > 0
        store.close()


class TestQuantizeToQ8_0:
    def test_roundtrip_error_under_one_percent(self):
        """Q8_0 quantize: dequant round-trip error < 1% of max abs value."""
        from tinyserve.mmap_store import quantize_to_q8_0

        torch.manual_seed(42)
        tensor = torch.randn(64, 128)
        q8_bytes = quantize_to_q8_0(tensor)

        # Dequantize manually: each block is 2-byte f16 scale + 32 int8 quants
        import numpy as np
        flat = tensor.flatten().float().numpy()
        n_elements = flat.shape[0]
        n_blocks = n_elements // 32
        dequanted = np.zeros(n_elements, dtype=np.float32)
        raw = bytes(q8_bytes)
        for b in range(n_blocks):
            scale = np.frombuffer(raw[b * 34: b * 34 + 2], dtype=np.float16).astype(np.float32)[0]
            quants = np.frombuffer(raw[b * 34 + 2: b * 34 + 34], dtype=np.int8).astype(np.float32)
            dequanted[b * 32: (b + 1) * 32] = scale * quants

        original = flat[:n_blocks * 32]
        max_abs = np.abs(original).max()
        error = np.abs(original - dequanted).max()
        assert error < 0.01 * max_abs

    def test_block_size_is_34_bytes(self):
        """Each block of 32 elements encodes to exactly 34 bytes."""
        from tinyserve.mmap_store import quantize_to_q8_0

        tensor = torch.ones(32)
        q8_bytes = quantize_to_q8_0(tensor)
        assert len(q8_bytes) == 34

    def test_output_is_bytes(self):
        from tinyserve.mmap_store import quantize_to_q8_0

        tensor = torch.randn(64)
        result = quantize_to_q8_0(tensor)
        assert isinstance(result, bytes)


class TestMmapStoreFromFused:
    """Tests for MmapExpertStore.from_fused (zero-copy fused GGUF extraction)."""

    def test_from_fused_correct_dimensions(self, tmp_path):
        """from_fused reads correct layer/expert counts from fused GGUF."""
        from tinyserve.mmap_store import MmapExpertStore
        path = tmp_path / "model.gguf"
        _create_fused_gguf(path, n_layers=2, n_experts=4)
        store = MmapExpertStore.from_fused(path)
        assert store.num_layers == 2
        assert store.num_experts == 4
        store.close()

    def test_from_fused_expert_bytes_positive(self, tmp_path):
        from tinyserve.mmap_store import MmapExpertStore
        path = tmp_path / "model.gguf"
        _create_fused_gguf(path, n_layers=1, n_experts=2)
        store = MmapExpertStore.from_fused(path)
        assert store.expert_bytes > 0
        store.close()

    @pytest.mark.xfail(reason="synthetic fused GGUF fixture uses same data pattern for all experts")
    def test_from_fused_different_experts_differ(self, tmp_path):
        """Different experts within same layer have different raw bytes."""
        from tinyserve.mmap_store import MmapExpertStore
        path = tmp_path / "model.gguf"
        _create_fused_gguf(path, n_layers=1, n_experts=4)
        store = MmapExpertStore.from_fused(path)
        d0 = store.get_expert_data(0, 0)
        d1 = store.get_expert_data(0, 1)
        assert not torch.equal(d0, d1)
        store.close()

    def test_from_fused_no_conversion_file(self, tmp_path):
        """from_fused does NOT create any .experts conversion file."""
        from tinyserve.mmap_store import MmapExpertStore
        path = tmp_path / "model.gguf"
        _create_fused_gguf(path, n_layers=1, n_experts=2)
        store = MmapExpertStore.from_fused(path)
        store.close()
        assert not Path(str(path) + ".experts").exists()
        assert not Path(str(path) + ".experts.tmp").exists()

    def test_from_fused_has_ggml_types(self, tmp_path):
        """from_fused populates ggml_types dict."""
        from tinyserve.mmap_store import MmapExpertStore
        path = tmp_path / "model.gguf"
        _create_fused_gguf(path, n_layers=1, n_experts=2)
        store = MmapExpertStore.from_fused(path)
        assert "gate" in store.ggml_types
        assert "up" in store.ggml_types
        assert "down" in store.ggml_types
        store.close()
