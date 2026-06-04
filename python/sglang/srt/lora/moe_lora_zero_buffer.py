"""MoE LoRA shrink-intermediate bump allocator (SGLANG_OPT_LORA_MOE_PREZERO).

dsv3-style ``gemm_output_zero_allocator`` analogue: one persistent buffer
(created in ``BaseLoRABackend.init_cuda_graph_moe_buffers``) is zeroed ONCE per
forward pass in ``LoRAManager.prepare_lora_batch`` — outside any cuda graph and
off the LoRA chain — and each MoE layer's gate_up-A / down-A split-K shrink
takes a fresh slice via ``take_zeroed_slice``. The per-call torch.zeros fills
(2 per MoE layer per step) disappear with NO cross-stream sync: graph/forward
launch ordering on the main stream already sequences the fill before all
consumers. Slices must be distinct per GEMM because each is an atomic_add
accumulation target.

Capacity shortfall (prefill-sized M, pointer exhaustion, dtype/device mismatch)
returns None and the caller falls back to its per-call torch.zeros path.

Torch-only on purpose: imported from the per-step ``prepare_lora_batch`` hook,
so it must not pull in triton.
"""

from __future__ import annotations

import torch

_MOE_LORA_ZERO_BUFFER: torch.Tensor | None = None
_MOE_LORA_ZERO_PTR: int = 0
_MOE_LORA_ZERO_MAX_SLICE: int = 0


def set_moe_lora_zero_buffer(
    buffer: torch.Tensor | None, max_slice_numel: int = 0
) -> None:
    """Install (or clear) the persistent pre-zeroed bump buffer.

    ``max_slice_numel`` is the per-GEMM slice budget the buffer was sized with
    (max_bs * top_k * max_lora_rank). Requests above it are refused (fallback)
    so a single forward can never consume more than the whole buffer — the
    invariant that makes the wraparound in ``take_zeroed_slice`` safe.
    """
    global _MOE_LORA_ZERO_BUFFER, _MOE_LORA_ZERO_PTR, _MOE_LORA_ZERO_MAX_SLICE
    _MOE_LORA_ZERO_BUFFER = buffer
    _MOE_LORA_ZERO_PTR = 0
    _MOE_LORA_ZERO_MAX_SLICE = max_slice_numel


def reset_moe_lora_zero_buffer() -> None:
    """Per-forward reset: re-zero the whole buffer (one fill kernel) and rewind
    the bump pointer. Runs on the eager, capture, AND graph-replay paths — so
    replayed graphs always see freshly zeroed slices. No-op when the buffer was
    never created (env off / no MoE LoRA)."""
    global _MOE_LORA_ZERO_PTR
    if _MOE_LORA_ZERO_BUFFER is not None:
        _MOE_LORA_ZERO_BUFFER.zero_()
        _MOE_LORA_ZERO_PTR = 0


def take_zeroed_slice(
    numel: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    """Bump-allocate a pre-zeroed slice; None if unavailable (caller falls back).

    Wraps to offset 0 when the tail is too small instead of failing: cuda-graph
    capture runs 2 warmup forwards + 1 captured forward after a SINGLE
    prepare_lora_batch pointer reset, so without wrapping the captured forward
    exhausts the buffer and bakes the per-call torch.zeros fallback into the
    graph (observed: zero engagement). Wrap safety: every request is capped at
    ``max_slice_numel`` and each forward issues at most (num_layers * 2)
    requests, so one forward never consumes more than the whole buffer — the
    post-wrap linear span therefore cannot overlap the same forward's pre-wrap
    slices (post-wrap span <= start offset). Stale warmup values don't matter:
    capture results are discarded and every replay/eager step re-zeroes the
    whole buffer first.
    """
    global _MOE_LORA_ZERO_PTR
    buf = _MOE_LORA_ZERO_BUFFER
    if (
        buf is None
        or buf.dtype != dtype
        or buf.device != device
        or numel > _MOE_LORA_ZERO_MAX_SLICE
    ):
        return None
    if _MOE_LORA_ZERO_PTR + numel > buf.numel():
        _MOE_LORA_ZERO_PTR = 0
    out = buf[_MOE_LORA_ZERO_PTR : _MOE_LORA_ZERO_PTR + numel]
    _MOE_LORA_ZERO_PTR += numel
    return out
