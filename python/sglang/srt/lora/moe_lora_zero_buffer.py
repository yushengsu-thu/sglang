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


def set_moe_lora_zero_buffer(buffer: torch.Tensor | None) -> None:
    """Install (or clear) the persistent pre-zeroed bump buffer."""
    global _MOE_LORA_ZERO_BUFFER, _MOE_LORA_ZERO_PTR
    _MOE_LORA_ZERO_BUFFER = buffer
    _MOE_LORA_ZERO_PTR = 0


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
    """Bump-allocate a pre-zeroed slice; None if unavailable (caller falls back)."""
    global _MOE_LORA_ZERO_PTR
    buf = _MOE_LORA_ZERO_BUFFER
    if (
        buf is None
        or buf.dtype != dtype
        or buf.device != device
        or _MOE_LORA_ZERO_PTR + numel > buf.numel()
    ):
        return None
    out = buf[_MOE_LORA_ZERO_PTR : _MOE_LORA_ZERO_PTR + numel]
    _MOE_LORA_ZERO_PTR += numel
    return out
