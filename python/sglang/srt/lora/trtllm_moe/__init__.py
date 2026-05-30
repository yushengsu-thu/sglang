"""Two-stream LoRA overlap (O1 + O7 + O8 + O9) — installed as a monkey-patch.

Activates when env ``SGLANG_LORA_TWO_STREAM=1``. Triggered exactly once via
:func:`install_two_stream_overrides` (called at end of ``sglang/srt/lora/layers.py``).

When enabled, these call sites are redirected to side-stream-overlapped versions
defined entirely in this package:

  * ``QKVParallelLinearWithLoRA.forward``  → :mod:`.attention.qkv_proj_lora_forward`
  * ``RowParallelLinearWithLoRA.forward``  → :mod:`.attention.row_parallel_lora_forward`
  * ``MergedColumnParallelLinearWithLoRA.forward`` → :mod:`.merged_column.merged_column_lora_forward`
  * ``fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora`` →
    :mod:`.moe_overlap.fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora_two_stream`

When disabled (env unset), ``install_two_stream_overrides`` is a no-op and all
the original functions / methods in ``sglang/srt/lora/layers.py`` and
``sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`` run unchanged.

Per-batch gating still happens inside the patched callables — they fall back
to the saved-original implementation for non-decode batches (token count above
``SGLANG_TWO_STREAM_MAX_TOKENS`` default 256), so prefill stays on the serial
path even with the patch installed.
"""
import os
from typing import Callable, List, Optional

import torch

from sglang.srt.environ import LoRABOverlapMode, envs

_ENV_KEY = "SGLANG_LORA_TWO_STREAM"
_MAX_TOKENS_KEY = "SGLANG_TWO_STREAM_MAX_TOKENS"
_MAX_TOKENS_DEFAULT = 256


def get_lora_b_overlap_mode() -> LoRABOverlapMode:
    """LoRA-B (expand) scheduling mode for the two-stream path."""
    return LoRABOverlapMode(envs.SGLANG_LORA_B_OVERLAP.get())


def is_two_stream_active(x: torch.Tensor) -> bool:
    """Per-batch gate. True iff env is on AND batch is decode-shaped."""
    if os.environ.get(_ENV_KEY) != "1":
        return False
    try:
        max_tok = int(os.environ.get(_MAX_TOKENS_KEY, str(_MAX_TOKENS_DEFAULT)))
    except ValueError:
        max_tok = _MAX_TOKENS_DEFAULT
    return x.shape[0] <= max_tok


_LORA_SIDE_STREAM: Optional[torch.cuda.Stream] = None


def get_lora_side_stream() -> torch.cuda.Stream:
    """Lazily allocate a single shared LoRA side stream.

    Within one decode layer the three sites (qkv → attn → o_proj → moe_gate_up)
    run sequentially, so one stream suffices and avoids extra graph-capture
    nodes from per-site streams.
    """
    global _LORA_SIDE_STREAM
    if _LORA_SIDE_STREAM is None:
        _LORA_SIDE_STREAM = torch.cuda.Stream()
    return _LORA_SIDE_STREAM


# Green-context streams for GREEN_CTX mode: (lora_partition_stream, base_partition_stream).
# Created once pre-capture; the two streams own disjoint SM partitions so the LoRA
# side-stream work and the SM-capped base gemm run truly concurrently.
_LORA_GREEN_STREAMS: Optional[tuple] = None
# (lora_sms, base_sms) actually granted by the green-ctx split; base_sms feeds
# DeepGEMM set_num_sms so the base gemm sizes its grid to its SM partition.
_LORA_GREEN_SM_SPLIT: Optional[tuple] = None


def get_lora_green_base_sms() -> Optional[int]:
    return _LORA_GREEN_SM_SPLIT[1] if _LORA_GREEN_SM_SPLIT else None


def get_lora_green_streams(
    device: Optional[torch.device] = None,
) -> Optional[tuple]:
    """Lazily allocate (lora_stream, base_stream) on disjoint SM partitions.

    Returns None if green contexts are unavailable (import/driver failure), so
    callers can fall back to the shared side stream.
    """
    global _LORA_GREEN_STREAMS, _LORA_GREEN_SM_SPLIT
    if _LORA_GREEN_STREAMS is not None:
        return _LORA_GREEN_STREAMS
    try:
        import flashinfer.green_ctx as gctx

        dev = device or torch.device("cuda", torch.cuda.current_device())
        total = torch.cuda.get_device_properties(dev).multi_processor_count
        # SM partitions must be a multiple of the green-ctx alignment (typically 8)
        # so the granted lora partition equals what we request and the remaining
        # base partition is known exactly.
        major, minor = gctx.get_compute_capability(dev)
        _min_sms, align = gctx.get_sm_count_constraint(major, minor)
        align = max(1, align)
        lora_sms = envs.SGLANG_LORA_GREEN_CTX_SMS.get()
        if lora_sms is None:
            lora_sms = total // 8
        lora_sms = max(align, (lora_sms // align) * align)
        # base gets the rest; DeepGEMM's set_num_sms requires an EVEN count.
        base_sms = max(2, total - lora_sms)
        base_sms -= base_sms % 2
        # streams[0] -> the requested lora partition; streams[-1] -> remaining SMs (base).
        streams, _res = gctx.split_device_green_ctx_by_sm_count(dev, [lora_sms])
        _LORA_GREEN_STREAMS = (streams[0], streams[-1])
        _LORA_GREEN_SM_SPLIT = (lora_sms, base_sms)
    except Exception as e:  # pragma: no cover - depends on driver/flashinfer
        import logging

        logging.getLogger(__name__).warning(
            "LoRA green-context split unavailable (%s); falling back to side stream.", e
        )
        _LORA_GREEN_STREAMS = None
    return _LORA_GREEN_STREAMS


def init_lora_two_stream_resources(device: Optional[torch.device] = None) -> None:
    """Eagerly create the side stream before cuda-graph capture begins.

    ``torch.cuda.Stream()`` is a driver call that must not run inside a
    cuda-graph capture region. Since :func:`get_lora_side_stream` is otherwise
    lazy, the first eligible decode forward would create it — which can fall
    inside capture if warmup didn't happen to exercise a two-stream batch.
    Calling this from a pre-capture hook pins creation to init/warmup on the
    correct device. No-op unless ``SGLANG_LORA_TWO_STREAM=1``.
    """
    if os.environ.get(_ENV_KEY) != "1":
        return
    if device is not None:
        with torch.cuda.device(device):
            get_lora_side_stream()
            if get_lora_b_overlap_mode() == LoRABOverlapMode.GREEN_CTX:
                get_lora_green_streams(device)
    else:
        get_lora_side_stream()
        if get_lora_b_overlap_mode() == LoRABOverlapMode.GREEN_CTX:
            get_lora_green_streams()


# References to the original implementations, captured at install time so the
# patched callables can defer to them for non-decode batches.
_ORIGINAL_QKV_FORWARD: Optional[Callable] = None
_ORIGINAL_ROW_FORWARD: Optional[Callable] = None
_ORIGINAL_MERGED_FORWARD: Optional[Callable] = None
_ORIGINAL_MOE_LORA_FUNC: Optional[Callable] = None
_INSTALLED: bool = False


def get_original_qkv_forward() -> Callable:
    return _ORIGINAL_QKV_FORWARD


def get_original_row_forward() -> Callable:
    return _ORIGINAL_ROW_FORWARD


def get_original_merged_column_forward() -> Callable:
    return _ORIGINAL_MERGED_FORWARD


def get_original_moe_lora_func() -> Callable:
    return _ORIGINAL_MOE_LORA_FUNC


def install_two_stream_overrides() -> None:
    """Install the side-stream overlapped overrides if ``SGLANG_LORA_TWO_STREAM=1``.

    Idempotent: subsequent calls are a no-op. Patches:

      1. ``QKVParallelLinearWithLoRA.forward`` (O7 — qkv LoRA shrink overlap)
      2. ``RowParallelLinearWithLoRA.forward`` (O8 — o_proj LoRA shrink overlap)
      3. ``MergedColumnParallelLinearWithLoRA.forward`` (O9 — merged-column LoRA
         shrink overlap: dense gate_up + mamba in_proj_qkvz)
      4. ``flashinfer_trtllm.fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora``
         (O1 — MoE gate_up LoRA overlap)

    The saved originals are exposed via :func:`get_original_qkv_forward`,
    :func:`get_original_row_forward`, :func:`get_original_moe_lora_func` so the
    new versions can fall back when their per-batch gate says single-stream.
    """
    global _INSTALLED, _ORIGINAL_QKV_FORWARD, _ORIGINAL_ROW_FORWARD, _ORIGINAL_MERGED_FORWARD, _ORIGINAL_MOE_LORA_FUNC

    if _INSTALLED:
        return
    if os.environ.get(_ENV_KEY) != "1":
        return

    from sglang.srt.lora.layers import (
        MergedColumnParallelLinearWithLoRA,
        QKVParallelLinearWithLoRA,
        RowParallelLinearWithLoRA,
    )
    from sglang.srt.lora.trtllm_moe.attention import (
        qkv_proj_lora_forward,
        row_parallel_lora_forward,
    )
    from sglang.srt.lora.trtllm_moe.merged_column import merged_column_lora_forward

    _ORIGINAL_QKV_FORWARD = QKVParallelLinearWithLoRA.forward
    _ORIGINAL_ROW_FORWARD = RowParallelLinearWithLoRA.forward
    _ORIGINAL_MERGED_FORWARD = MergedColumnParallelLinearWithLoRA.forward
    QKVParallelLinearWithLoRA.forward = qkv_proj_lora_forward
    RowParallelLinearWithLoRA.forward = row_parallel_lora_forward
    MergedColumnParallelLinearWithLoRA.forward = merged_column_lora_forward

    import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm as ft
    from sglang.srt.lora.trtllm_moe.moe_overlap import (
        fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora_two_stream,
    )

    _ORIGINAL_MOE_LORA_FUNC = ft.fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora
    ft.fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora = (
        fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora_two_stream
    )

    _INSTALLED = True


__all__ = [
    "is_two_stream_active",
    "get_lora_side_stream",
    "get_lora_green_streams",
    "get_lora_b_overlap_mode",
    "init_lora_two_stream_resources",
    "get_original_qkv_forward",
    "get_original_row_forward",
    "get_original_merged_column_forward",
    "get_original_moe_lora_func",
    "install_two_stream_overrides",
]
