# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit tests for MoE LoRA TP weight slicing and buffer shape correctness.

These tests run on a single GPU by simulating multiple TP ranks.
They verify that:
1. slice methods produce correct shapes
2. sliced weights across all ranks reconstruct the original
3. buffer shapes from mem_pool match the sliced weight shapes
4. per-rank MoE+LoRA forward produces results consistent with the full-weight forward
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.distributed import divide
from sglang.srt.lora.utils import (
    ROW_PARALLELISM_LINEAR_LORA_NAMES,
    get_hidden_dim,
    get_stacked_multiply,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fused_moe_with_lora_mock(
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    tp_size: int,
    tp_rank: int,
):
    """Build a lightweight mock that has the same slice methods as FusedMoEWithLoRA."""
    import types

    from sglang.srt.lora.layers import FusedMoEWithLoRA

    base = SimpleNamespace(
        moe_tp_size=tp_size,
        moe_tp_rank=tp_rank,
        num_experts=num_experts,
        intermediate_size_per_partition=intermediate_size // tp_size,
        hidden_size=hidden_size,
    )

    class _Mock:
        pass

    obj = _Mock()
    obj.base_layer = base
    obj.tp_size = tp_size
    obj.tp_rank = tp_rank

    for name in (
        "slice_w13_lora_a",
        "slice_w13_lora_b",
        "slice_w2_lora_a",
        "slice_w2_lora_b",
    ):
        setattr(obj, name, types.MethodType(getattr(FusedMoEWithLoRA, name), obj))
    return obj


# ===========================================================================
# Test 1: slice methods — shapes
# ===========================================================================


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("num_experts", [8, 16])
@pytest.mark.parametrize("lora_rank", [8, 16])
def test_slice_shapes(tp_size, num_experts, lora_rank):
    hidden_size = 512
    intermediate_size = 1024

    for tp_rank in range(tp_size):
        m = _make_fused_moe_with_lora_mock(
            num_experts, intermediate_size, hidden_size, tp_size, tp_rank
        )
        inter_part = intermediate_size // tp_size

        # gate_up A: (rank*2, hidden_size) -> unchanged
        A13 = torch.randn(lora_rank * 2, hidden_size)
        out = m.slice_w13_lora_a(A13, tp_rank)
        assert out.shape == (lora_rank * 2, hidden_size)

        # gate_up B: (intermediate_size*2, rank) -> (inter_part*2, rank)
        B13 = torch.randn(intermediate_size * 2, lora_rank)
        out = m.slice_w13_lora_b(B13, tp_rank)
        assert out.shape == (inter_part * 2, lora_rank), (
            f"tp_size={tp_size}, rank={tp_rank}: "
            f"expected {(inter_part * 2, lora_rank)}, got {out.shape}"
        )

        # down A: (rank, intermediate_size) -> (rank, inter_part)
        A2 = torch.randn(lora_rank, intermediate_size)
        out = m.slice_w2_lora_a(A2, tp_rank)
        assert out.shape == (lora_rank, inter_part), (
            f"tp_size={tp_size}, rank={tp_rank}: "
            f"expected {(lora_rank, inter_part)}, got {out.shape}"
        )

        # down B: (hidden_size, rank) -> unchanged
        B2 = torch.randn(hidden_size, lora_rank)
        out = m.slice_w2_lora_b(B2, tp_rank)
        assert out.shape == (hidden_size, lora_rank)


# ===========================================================================
# Test 2: slice methods — reconstruction
# ===========================================================================


@pytest.mark.parametrize("tp_size", [2, 4])
def test_slice_reconstruction(tp_size):
    """Sliced weights from all ranks, when concatenated, should reconstruct the original."""
    hidden_size = 256
    intermediate_size = 512
    lora_rank = 8
    num_experts = 8

    # ---- gate_up B reconstruction ----
    B13_full = torch.randn(intermediate_size * 2, lora_rank)
    gate_full = B13_full[:intermediate_size, :]
    up_full = B13_full[intermediate_size:, :]

    gate_parts = []
    up_parts = []
    for tp_rank in range(tp_size):
        m = _make_fused_moe_with_lora_mock(
            num_experts, intermediate_size, hidden_size, tp_size, tp_rank
        )
        sliced = m.slice_w13_lora_b(B13_full.clone(), tp_rank)
        half = sliced.shape[0] // 2
        gate_parts.append(sliced[:half, :])
        up_parts.append(sliced[half:, :])

    gate_reconstructed = torch.cat(gate_parts, dim=0)
    up_reconstructed = torch.cat(up_parts, dim=0)

    torch.testing.assert_close(gate_reconstructed, gate_full)
    torch.testing.assert_close(up_reconstructed, up_full)

    # ---- down A reconstruction ----
    A2_full = torch.randn(lora_rank, intermediate_size)
    parts = []
    for tp_rank in range(tp_size):
        m = _make_fused_moe_with_lora_mock(
            num_experts, intermediate_size, hidden_size, tp_size, tp_rank
        )
        parts.append(m.slice_w2_lora_a(A2_full.clone(), tp_rank))

    reconstructed = torch.cat(parts, dim=1)
    torch.testing.assert_close(reconstructed, A2_full)


# ===========================================================================
# Test 3: buffer shape computation in LoRAMemoryPool
# ===========================================================================


def _is_moe_module(self, module_name: str) -> bool:
    return "moe" in module_name


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_buffer_shapes(tp_size):
    """get_lora_A_shape / get_lora_B_shape should match the expected TP-partitioned shapes."""
    hidden_size = 512
    moe_intermediate_size = 1024
    num_experts = 8
    max_lora_rank = 16
    max_loras = 4

    class MockConfig:
        pass

    config = MockConfig()
    config.hidden_size = hidden_size
    config.intermediate_size = moe_intermediate_size
    config.moe_intermediate_size = moe_intermediate_size
    config.num_experts = num_experts
    config.num_attention_heads = 16
    config.num_key_value_heads = 4

    class MockModel:
        pass

    model = MockModel()
    model.config = config

    from sglang.srt.lora.mem_pool import LoRAMemoryPool

    mock_self = SimpleNamespace(
        base_hf_config=config,
        tp_size=tp_size,
        max_loras_per_batch=max_loras,
        is_moe_module=_is_moe_module.__get__(None, type(None)),
    )
    mock_self.is_moe_module = lambda name: "moe" in name

    inter_part = moe_intermediate_size // tp_size

    for module_name, expected_a, expected_b in [
        (
            "gate_up_proj_moe",
            (max_loras, num_experts, max_lora_rank * 2, hidden_size),
            (max_loras, num_experts, inter_part * 2, max_lora_rank),
        ),
        (
            "down_proj_moe",
            (max_loras, num_experts, max_lora_rank, inter_part),
            (max_loras, num_experts, hidden_size, max_lora_rank),
        ),
    ]:
        a_shape = LoRAMemoryPool.get_lora_A_shape(
            mock_self, module_name, model, max_lora_rank, 0
        )
        b_shape = LoRAMemoryPool.get_lora_B_shape(
            mock_self, module_name, model, max_lora_rank, 0
        )
        assert a_shape == expected_a, (
            f"[{module_name}] A shape: expected {expected_a}, got {a_shape}"
        )
        assert b_shape == expected_b, (
            f"[{module_name}] B shape: expected {expected_b}, got {b_shape}"
        )


# ===========================================================================
# Test 4: buffer shape matches sliced weight shape (per-expert)
# ===========================================================================


@pytest.mark.parametrize("tp_size", [2, 4])
def test_sliced_weight_fits_buffer(tp_size):
    """Per-expert sliced weights should fit exactly into the buffer allocated by get_lora_*_shape."""
    hidden_size = 512
    intermediate_size = 1024
    lora_rank = 8
    num_experts = 8

    inter_part = intermediate_size // tp_size

    for tp_rank in range(tp_size):
        m = _make_fused_moe_with_lora_mock(
            num_experts, intermediate_size, hidden_size, tp_size, tp_rank
        )

        # gate_up A: per-expert buffer slot is (max_rank*2, hidden_size)
        A13 = torch.randn(lora_rank * 2, hidden_size)
        sliced = m.slice_w13_lora_a(A13, tp_rank)
        assert sliced.shape == (lora_rank * 2, hidden_size)

        # gate_up B: per-expert buffer slot is (inter_part*2, max_rank)
        B13 = torch.randn(intermediate_size * 2, lora_rank)
        sliced = m.slice_w13_lora_b(B13, tp_rank)
        assert sliced.shape == (inter_part * 2, lora_rank)

        # down A: per-expert buffer slot is (max_rank, inter_part)
        A2 = torch.randn(lora_rank, intermediate_size)
        sliced = m.slice_w2_lora_a(A2, tp_rank)
        assert sliced.shape == (lora_rank, inter_part)

        # down B: per-expert buffer slot is (hidden_size, max_rank)
        B2 = torch.randn(hidden_size, lora_rank)
        sliced = m.slice_w2_lora_b(B2, tp_rank)
        assert sliced.shape == (hidden_size, lora_rank)


# ===========================================================================
# Test 5: down_proj_moe is in ROW_PARALLELISM_LINEAR_LORA_NAMES
# ===========================================================================


def test_down_proj_moe_in_row_parallelism():
    assert "down_proj_moe" in ROW_PARALLELISM_LINEAR_LORA_NAMES
    assert "down_proj" in ROW_PARALLELISM_LINEAR_LORA_NAMES


# ===========================================================================
# Test 6: TP correctness — per-rank gate_up LoRA delta
# ===========================================================================


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("lora_rank", [4, 8])
def test_tp_lora_delta_gate_up(tp_size, lora_rank):
    """For gate_up (column-parallel), each TP rank computes a shard of the output.
    Gathering gate shards and up shards separately, then concatenating, should
    equal the full LoRA delta."""
    hidden_size = 128
    intermediate_size = 256
    num_experts = 4
    inter_part = intermediate_size // tp_size

    torch.manual_seed(42)
    x = torch.randn(1, hidden_size)

    A_full = torch.randn(lora_rank * 2, hidden_size)
    B_full = torch.randn(intermediate_size * 2, lora_rank)
    r = lora_rank

    gate_a = A_full[:r, :]
    up_a = A_full[r:, :]
    half = B_full.shape[0] // 2
    gate_b = B_full[:half, :]
    up_b = B_full[half:, :]

    gate_delta_full = (gate_b @ (gate_a @ x.T)).T  # (1, intermediate_size)
    up_delta_full = (up_b @ (up_a @ x.T)).T  # (1, intermediate_size)
    full_delta = torch.cat([gate_delta_full, up_delta_full], dim=1)

    gate_shards = []
    up_shards = []
    for tp_rank in range(tp_size):
        m = _make_fused_moe_with_lora_mock(
            num_experts, intermediate_size, hidden_size, tp_size, tp_rank
        )
        A_sliced = m.slice_w13_lora_a(A_full.clone(), tp_rank)
        B_sliced = m.slice_w13_lora_b(B_full.clone(), tp_rank)

        sh = B_sliced.shape[0] // 2
        gate_a_s = A_sliced[:r, :]
        up_a_s = A_sliced[r:, :]
        gate_b_s = B_sliced[:sh, :]
        up_b_s = B_sliced[sh:, :]

        gate_shards.append((gate_b_s @ (gate_a_s @ x.T)).T)  # (1, inter_part)
        up_shards.append((up_b_s @ (up_a_s @ x.T)).T)  # (1, inter_part)

    # Column-parallel: gather output shards, then concatenate [gate || up]
    reconstructed = torch.cat(
        [torch.cat(gate_shards, dim=1), torch.cat(up_shards, dim=1)], dim=1
    )
    torch.testing.assert_close(reconstructed, full_delta, atol=1e-4, rtol=1e-4)


# ===========================================================================
# Test 7: TP correctness — per-rank down LoRA delta
# ===========================================================================


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("lora_rank", [4, 8])
def test_tp_lora_delta_down(tp_size, lora_rank):
    """For down (row-parallel), each TP rank operates on a partition of the input.
    Summing (all-reduce) all rank outputs should equal the full LoRA delta."""
    hidden_size = 128
    intermediate_size = 256
    num_experts = 4
    inter_part = intermediate_size // tp_size

    torch.manual_seed(42)
    x_full = torch.randn(1, intermediate_size)

    A_full = torch.randn(lora_rank, intermediate_size)
    B_full = torch.randn(hidden_size, lora_rank)
    full_delta = (B_full @ (A_full @ x_full.T)).T

    rank_outputs = []
    for tp_rank in range(tp_size):
        m = _make_fused_moe_with_lora_mock(
            num_experts, intermediate_size, hidden_size, tp_size, tp_rank
        )
        A_sliced = m.slice_w2_lora_a(A_full.clone(), tp_rank)
        B_sliced = m.slice_w2_lora_b(B_full.clone(), tp_rank)
        x_part = x_full[:, tp_rank * inter_part : (tp_rank + 1) * inter_part]
        rank_delta = (B_sliced @ (A_sliced @ x_part.T)).T
        rank_outputs.append(rank_delta)

    summed = sum(rank_outputs)
    torch.testing.assert_close(summed, full_delta, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
