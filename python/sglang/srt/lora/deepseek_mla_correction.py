"""LoRA correction for absorbed-MLA ``kv_b_proj``.

The absorbed-MLA path in ``DeepseekV2AttentionMLA`` bypasses
``kv_b_proj.forward()`` and folds the K/V contribution into two BMMs against
the pre-computed ``w_kc`` / ``w_vc`` weights, so a standard
``ColumnParallelLinearWithLoRA`` wrapper would never see the activations and
the LoRA delta would silently be dropped. These helpers inject the missing
delta on top of the absorbed intermediates via the SGMM-style Triton kernels
in ``triton_ops/kv_b_lora_absorbed.py``.

Used from ``deepseek_common/attention_forward_methods/forward_mla.py``. Call
sites should gate the call with :func:`is_kv_b_lora_active` so non-LoRA
forwards take a single ``getattr`` and skip the helper entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.lora.triton_ops import (
    step_a_q_fwd,
    step_a_v_fwd,
    step_b_q_fwd,
    step_b_v_fwd,
)

if TYPE_CHECKING:
    from sglang.srt.lora.utils import LoRABatchInfo
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA


# ---------------------------------------------------------------------------
# Experimental dense (standard torch bmm/matmul) path, gated by
# ``SGLANG_OPT_MLA_LORA_DENSE_GEMM``. Replaces the per-head Triton SGMM step
# kernels with batched dense gemms for the single-active-adapter decode case.
# The four step kernels are tiny per-head SGMMs (rank ~16-32) and profiling
# flagged them as a non-overlapped ~45us hotspot; the question is whether a
# plain bmm (cuBLAS) beats the fused-but-tiny Triton tiles. See journal.
# ---------------------------------------------------------------------------


def _dense_path_active(batch_info: "LoRABatchInfo") -> bool:
    """Whether the experimental dense kv_b LoRA correction applies.

    Restricted to the single-active-adapter case so we can gather one slot's
    weights graph-safely (length-1 ``index_select``, no ``.item()`` sync) and
    skip per-segment routing. ``num_segments`` is a host-side int; for the
    Triton backend single-adapter decode repeats merge into one segment. A row
    permutation means adapter-grouped chunking (multi-adapter) -> fall back to
    the Triton kernels. Assumes buffer rank == active adapter rank (the common
    single-adapter deployment); mixed/zero-padded ranks would need a per-slot
    truncation that costs a GPU sync, so those keep the kernel path.
    """
    if not envs.SGLANG_OPT_MLA_LORA_DENSE_GEMM.get():
        return False
    if batch_info.permutation is not None:
        return False
    return (batch_info.num_segments or batch_info.bs) == 1


def _slot_weights(A_buf, B_buf, batch_info):
    """Gather the single active slot's ``(A, B, scaling)``. Graph-safe."""
    slot = batch_info.weight_indices[:1]  # (1,)
    A = A_buf.index_select(0, slot).squeeze(0)  # (rank, kv_lora_rank)
    B = B_buf.index_select(0, slot).squeeze(0)  # (H*FULL_K, rank)
    scaling = batch_info.scalings.index_select(0, slot)  # (1,) tensor
    return A, B, scaling


def _dense_step_a_q(q_nope, B, full_K_per_head):
    """step A_q as a batched gemm: ``(S,H,qk_nope) -> (H,S,rank)``."""
    S, H, qk = q_nope.shape
    rank = B.shape[-1]
    B_kc = B.view(H, full_K_per_head, rank)[:, :qk, :].contiguous()  # (H,qk,rank)
    return torch.bmm(q_nope.transpose(0, 1).contiguous(), B_kc)  # (H,S,rank)


def _dense_step_b_q(q_lora_a_hsr, A, scaling, q_nope_out):
    """step B_q: ``q_nope_out (S,H,kv) += (H,S,rank) @ A(rank,kv) * scaling``."""
    delta = torch.matmul(q_lora_a_hsr, A * scaling)  # (H,S,kv)
    q_nope_out.add_(delta.transpose(0, 1))
    return q_nope_out


def _dense_step_a_v(attn_output, A):
    """step A_v as a batched gemm: ``(S,H,kv) @ A.T(kv,rank) -> (H,S,rank)``."""
    return torch.matmul(attn_output.transpose(0, 1).contiguous(), A.t())


def _dense_step_b_v(attn_lora_a_hsr, B, scaling, base_view, qk_nope, v_head_dim):
    """step B_v: ``base (S,H,v) += (H,S,rank) @ B_vc.T(H,rank,v) * scaling``."""
    H = base_view.shape[1]
    rank = B.shape[-1]
    full_K = qk_nope + v_head_dim
    B_vc = B.view(H, full_K, rank)[:, qk_nope:, :].contiguous()  # (H,v,rank)
    delta = torch.bmm(attn_lora_a_hsr, (B_vc * scaling).transpose(1, 2))  # (H,S,v)
    base_view.add_(delta.transpose(0, 1))
    return base_view


def is_kv_b_lora_active(attn_module: "DeepseekV2AttentionMLA") -> bool:
    """Cheap precondition check used at call sites in the attention forward
    to skip the entire LoRA-correction path when no ``kv_b_proj`` adapter is
    wrapped on this module (the common case)."""
    return getattr(attn_module.kv_b_proj, "set_lora", False)


def _get_state(
    attn_module: "DeepseekV2AttentionMLA",
) -> Optional[Tuple[torch.Tensor, torch.Tensor, "LoRABatchInfo"]]:
    if not is_kv_b_lora_active(attn_module):
        return None
    if not hasattr(attn_module.kv_b_proj, "A_buffer"):
        return None
    lora_backend = attn_module.kv_b_proj.lora_backend
    if not hasattr(lora_backend, "batch_info"):
        return None
    batch_info = lora_backend.batch_info
    if batch_info is None:
        return None

    # Triton backend exposes _sgemm_info() to group decode-shape repeats of
    # the same adapter; csgmv-style backends just expose batch_info directly.
    sgemm_info = getattr(lora_backend, "_sgemm_info", None)
    if callable(sgemm_info):
        batch_info = sgemm_info()
    return attn_module.kv_b_proj.A_buffer, attn_module.kv_b_proj.B_buffer, batch_info


def apply_q_correction(
    attn_module: "DeepseekV2AttentionMLA",
    q_nope: torch.Tensor,
    q_nope_out: torch.Tensor,
) -> torch.Tensor:
    """LoRA correction for the absorbed ``q_nope @ w_kc`` path.

    Computes ``q_nope_out += q_nope @ B_kc @ A * scaling`` per token, per
    active LoRA slot via two SGMM-style Triton kernels. Factored along the
    LoRA-A/B boundary so we never materialise ``B @ A`` (~268M FMAs per layer
    per slot in the naive implementation)::

      step A_q : ``(S,H,qk_nope) @ B_kc[slot, h] (qk_nope, rank) -> (S,H,rank)``
      step B_q : ``(S,H,rank)    @ A[slot] (rank, kv_lora_rank)  -> += q_nope_out``
    """
    state = _get_state(attn_module)
    if state is None:
        return q_nope_out
    A_buf, B_buf, batch_info = state

    full_K_per_head = attn_module.qk_nope_head_dim + attn_module.v_head_dim
    if _dense_path_active(batch_info):
        A, B, scaling = _slot_weights(A_buf, B_buf, batch_info)
        q_lora_a = _dense_step_a_q(q_nope, B, full_K_per_head)
        return _dense_step_b_q(q_lora_a, A, scaling, q_nope_out)
    q_lora_a = step_a_q_fwd(q_nope, B_buf, batch_info, full_K_per_head)
    return step_b_q_fwd(q_lora_a, A_buf, batch_info, q_nope_out)


def apply_v_correction(
    attn_module: "DeepseekV2AttentionMLA",
    attn_output: torch.Tensor,
    attn_bmm_flat: torch.Tensor,
) -> torch.Tensor:
    """LoRA correction for the absorbed ``attn_output @ w_vc`` path.

    Computes ``attn_bmm_flat += attn_output @ A.T @ B_vc.T * scaling`` per
    token, per active LoRA slot. ``attn_bmm_flat`` is the flat
    ``(S, H*v_head_dim)`` view of the absorbed BMM result; we pass strides
    matching the implicit ``(S, H, v_head_dim)`` layout to step B_v.
    """
    state = _get_state(attn_module)
    if state is None:
        return attn_bmm_flat
    A_buf, B_buf, batch_info = state

    base_view = attn_bmm_flat.view(
        -1, attn_module.num_local_heads, attn_module.v_head_dim
    )
    if _dense_path_active(batch_info):
        A, B, scaling = _slot_weights(A_buf, B_buf, batch_info)
        attn_lora_a = _dense_step_a_v(attn_output, A)
        _dense_step_b_v(
            attn_lora_a,
            B,
            scaling,
            base_view,
            attn_module.qk_nope_head_dim,
            attn_module.v_head_dim,
        )
        return attn_bmm_flat

    attn_lora_a = step_a_v_fwd(attn_output, A_buf, batch_info)
    step_b_v_fwd(
        attn_lora_a,
        B_buf,
        batch_info,
        base_view,
        attn_module.qk_nope_head_dim,
        attn_module.v_head_dim,
    )
    return attn_bmm_flat


# ---------------------------------------------------------------------------
# Two-stream overlap (O12) for the absorbed kv_b correction.
#
# Each correction factors into an input-only A-step (reads q_nope / attn_output,
# independent of the absorbed bmm output) and a B-step that adds into that bmm
# output. ``*_prepare`` forks the A-step onto the shared LoRA side stream so it
# overlaps the main-stream ``q_nope @ w_kc`` / ``attn_output @ w_vc`` bmm;
# ``*_apply`` rejoins and runs the B-step.
#
# Gated by ``SGLANG_LORA_TWO_STREAM`` (decode batches only) via
# ``is_two_stream_active``. When inactive, ``*_prepare`` returns None and
# ``*_apply`` falls back to the serial ``apply_*_correction`` (or a no-op when no
# kv_b adapter is wrapped), so the deepseek call sites stay byte-identical with
# two-stream off. Same fork/join (``wait_stream``) idiom as the O7/O8 attention
# overrides — cuda-graph-capture safe.
# ---------------------------------------------------------------------------


def _kv_b_two_stream_state(attn_module, x):
    from sglang.srt.lora.trtllm_moe import get_lora_side_stream, is_two_stream_active

    if not is_two_stream_active(x):
        return None
    state = _get_state(attn_module)
    if state is None:
        return None
    A_buf, B_buf, batch_info = state
    return A_buf, B_buf, batch_info, get_lora_side_stream()


def kv_b_lora_q_prepare(attn_module, q_nope):
    """Fork the q-correction A-step onto the side stream (``step_a_q`` reads only
    ``q_nope``) so it overlaps the main-stream ``q_nope @ w_kc`` bmm. Returns a
    handle for :func:`kv_b_lora_q_apply`, or None when two-stream is inactive."""
    st = _kv_b_two_stream_state(attn_module, q_nope)
    if st is None:
        return None
    A_buf, B_buf, batch_info, side_stream = st
    full_K_per_head = attn_module.qk_nope_head_dim + attn_module.v_head_dim
    side_stream.wait_stream(torch.cuda.current_stream())
    if _dense_path_active(batch_info):
        A, B, scaling = _slot_weights(A_buf, B_buf, batch_info)
        with torch.cuda.stream(side_stream):
            q_lora_a = _dense_step_a_q(q_nope, B, full_K_per_head)
        return "dense_q", q_lora_a, A, scaling, side_stream
    with torch.cuda.stream(side_stream):
        q_lora_a = step_a_q_fwd(q_nope, B_buf, batch_info, full_K_per_head)
    return q_lora_a, A_buf, batch_info, side_stream


def kv_b_lora_q_apply(attn_module, q_nope, q_nope_out, handle):
    """Finish the q-correction: two-stream (rejoin + B-step) when ``handle`` is
    set, else the serial correction, else a no-op. Single call replacing the
    ``if is_kv_b_lora_active: apply_q_correction`` at the call site."""
    if handle is not None:
        if isinstance(handle[0], str):  # dense marker
            _, q_lora_a, A, scaling, side_stream = handle
            torch.cuda.current_stream().wait_stream(side_stream)
            return _dense_step_b_q(q_lora_a, A, scaling, q_nope_out)
        q_lora_a, A_buf, batch_info, side_stream = handle
        torch.cuda.current_stream().wait_stream(side_stream)
        return step_b_q_fwd(q_lora_a, A_buf, batch_info, q_nope_out)
    if is_kv_b_lora_active(attn_module):
        return apply_q_correction(attn_module, q_nope, q_nope_out)
    return q_nope_out


def kv_b_lora_v_prepare(attn_module, attn_output):
    """Fork the v-correction A-step onto the side stream (``step_a_v`` reads only
    ``attn_output``) so it overlaps the main-stream ``attn_output @ w_vc`` bmm.
    Returns a handle for :func:`kv_b_lora_v_apply`, or None when inactive."""
    st = _kv_b_two_stream_state(attn_module, attn_output)
    if st is None:
        return None
    A_buf, B_buf, batch_info, side_stream = st
    side_stream.wait_stream(torch.cuda.current_stream())
    if _dense_path_active(batch_info):
        A, B, scaling = _slot_weights(A_buf, B_buf, batch_info)
        with torch.cuda.stream(side_stream):
            attn_lora_a = _dense_step_a_v(attn_output, A)
        return "dense_v", attn_lora_a, B, scaling, side_stream
    with torch.cuda.stream(side_stream):
        attn_lora_a = step_a_v_fwd(attn_output, A_buf, batch_info)
    return attn_lora_a, B_buf, batch_info, side_stream


def kv_b_lora_v_apply(attn_module, attn_output, attn_bmm_flat, handle):
    """Finish the v-correction: two-stream (rejoin + B-step) when ``handle`` is
    set, else the serial correction, else a no-op."""
    if handle is not None:
        base_view = attn_bmm_flat.view(
            -1, attn_module.num_local_heads, attn_module.v_head_dim
        )
        if isinstance(handle[0], str):  # dense marker
            _, attn_lora_a, B, scaling, side_stream = handle
            torch.cuda.current_stream().wait_stream(side_stream)
            _dense_step_b_v(
                attn_lora_a,
                B,
                scaling,
                base_view,
                attn_module.qk_nope_head_dim,
                attn_module.v_head_dim,
            )
            return attn_bmm_flat
        attn_lora_a, B_buf, batch_info, side_stream = handle
        torch.cuda.current_stream().wait_stream(side_stream)
        step_b_v_fwd(
            attn_lora_a,
            B_buf,
            batch_info,
            base_view,
            attn_module.qk_nope_head_dim,
            attn_module.v_head_dim,
        )
        return attn_bmm_flat
    if is_kv_b_lora_active(attn_module):
        return apply_v_correction(attn_module, attn_output, attn_bmm_flat)
    return attn_bmm_flat
