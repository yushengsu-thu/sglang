"""Single-LoRA FUSED absorbed-MLA kv_b correction kernels.

For the single-active-adapter case (max_loras_per_batch==1) the multi-LoRA machinery in
``kv_b_lora_absorbed.py`` (weight_indices / lora_ranks / N_eff truncation / seg routing /
permutation gather / segment grid axis) all degenerates to constants. This module collapses each
correction's two step kernels into ONE fused kernel that keeps the tiny rank-dim intermediate in
registers (no HBM round-trip) — a plain tiled gemm with a low-rank bottleneck fused in.

q-correction:  q_nope_out[s,h,k] += scaling * ( q_nope[s,h,:] @ B_kc[h] ) @ A         (A_q -> B_q)
v-correction:  attn_bmm[s,h,j]  += scaling * ( attn[s,h,:] @ A.T ) @ B_vc[h].T         (A_v -> B_v)

  B (single slot): (H*FULL_K, R), FULL_K = qk_nope + v_head_dim; B_kc = rows [h*FULL_K : +qk_nope],
                   B_vc = rows [h*FULL_K+qk_nope : +v_head_dim].
  A (single slot): (R, kv_lora_rank).

Grid = (cdiv(S, BLOCK_S), H). Everything else constexpr. Caller must guarantee single slot.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_S = 32
_BLOCK_K = 64
_BLOCK_N = 128


@triton.jit
def _fused_q_kernel(
    x, B, A, out,
    S, scaling_ptr,  # 1-elem fp32 tensor (graph-safe: loaded on device, never .item())
    x_s, x_h, x_k,
    b_row, b_r,
    a_r, a_n,
    o_s, o_h, o_n,
    H: tl.constexpr, QK: tl.constexpr, KV: tl.constexpr, R: tl.constexpr,
    FULL_K: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    head = tl.program_id(1)
    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S
    r_off = tl.arange(0, R)
    scaling = tl.load(scaling_ptr).to(tl.float32)

    # --- step A_q: lora_a[BLOCK_S, R] = x[s,h,:] @ B_kc[h]  (contract QK) ---
    acc_a = tl.zeros((BLOCK_S, R), dtype=tl.float32)
    head_base = head * FULL_K  # K-half rows start here
    for k0 in range(0, QK, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        xt = tl.load(
            x + s_off[:, None] * x_s + head * x_h + k_off[None, :] * x_k,
            mask=s_mask[:, None], other=0.0,
        )  # (BLOCK_S, BLOCK_K)
        bt = tl.load(
            B + (head_base + k_off[:, None]) * b_row + r_off[None, :] * b_r
        )  # (BLOCK_K, R)
        acc_a += tl.dot(xt, bt)
    lora_a = acc_a.to(x.dtype.element_ty)  # (BLOCK_S, R)

    # --- step B_q: out[s,h,n] += scaling * lora_a @ A  (contract R, n over KV) ---
    for n0 in range(0, KV, BLOCK_N):
        n_off = n0 + tl.arange(0, BLOCK_N)
        at = tl.load(A + r_off[:, None] * a_r + n_off[None, :] * a_n)  # (R, BLOCK_N)
        acc_o = tl.dot(lora_a, at) * scaling  # (BLOCK_S, BLOCK_N) fp32
        op = out + s_off[:, None] * o_s + head * o_h + n_off[None, :] * o_n
        prev = tl.load(op, mask=s_mask[:, None], other=0.0)
        tl.store(op, prev + acc_o.to(out.dtype.element_ty), mask=s_mask[:, None])


@triton.jit
def _fused_v_kernel(
    x, A, B, out,
    S, scaling_ptr,  # 1-elem fp32 tensor (graph-safe)
    x_s, x_h, x_k,
    a_r, a_k,
    b_row, b_r,
    o_s, o_h, o_n,
    H: tl.constexpr, KV: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    FULL_K: tl.constexpr, QK_OFF: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    head = tl.program_id(1)
    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S
    r_off = tl.arange(0, R)
    scaling = tl.load(scaling_ptr).to(tl.float32)

    # --- step A_v: lora_a[BLOCK_S, R] = x[s,h,:] @ A.T  (contract KV) ---
    acc_a = tl.zeros((BLOCK_S, R), dtype=tl.float32)
    for k0 in range(0, KV, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        xt = tl.load(
            x + s_off[:, None] * x_s + head * x_h + k_off[None, :] * x_k,
            mask=s_mask[:, None], other=0.0,
        )  # (BLOCK_S, BLOCK_K)
        akt = tl.load(A + r_off[None, :] * a_r + k_off[:, None] * a_k)  # (BLOCK_K, R) = A[:,k].T
        acc_a += tl.dot(xt, akt)
    lora_a = acc_a.to(x.dtype.element_ty)  # (BLOCK_S, R)

    # --- step B_v: out[s,h,j] += scaling * lora_a @ B_vc[h].T  (contract R, j over V) ---
    head_base = head * FULL_K + QK_OFF  # V-half rows
    for n0 in range(0, V, BLOCK_N):
        n_off = n0 + tl.arange(0, BLOCK_N)
        bt = tl.load(
            B + (head_base + n_off[None, :]) * b_row + r_off[:, None] * b_r
        )  # (R, BLOCK_N) = B_vc[h].T
        acc_o = tl.dot(lora_a, bt) * scaling  # (BLOCK_S, BLOCK_N)
        op = out + s_off[:, None] * o_s + head * o_h + n_off[None, :] * o_n
        prev = tl.load(op, mask=s_mask[:, None], other=0.0)
        tl.store(op, prev + acc_o.to(out.dtype.element_ty), mask=s_mask[:, None])


def _scaling_tensor(scaling, device):
    """Accept a 1-elem device tensor (graph-safe path) or a python float (tests)."""
    if torch.is_tensor(scaling):
        return scaling
    return torch.tensor([scaling], dtype=torch.float32, device=device)


def fused_q_correction(q_nope, B, A, scaling, q_nope_out):
    """q_nope (S,H,QK); B (H*FULL_K,R); A (R,KV); scaling: 1-elem tensor or float;
    q_nope_out (S,H,KV) in-place."""
    S, H, QK = q_nope.shape
    R = B.shape[-1]
    KV = A.shape[-1]
    FULL_K = B.shape[0] // H
    grid = (triton.cdiv(S, _BLOCK_S), H)
    _fused_q_kernel[grid](
        q_nope, B, A, q_nope_out, S, _scaling_tensor(scaling, q_nope.device),
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        B.stride(0), B.stride(1), A.stride(0), A.stride(1),
        q_nope_out.stride(0), q_nope_out.stride(1), q_nope_out.stride(2),
        H=H, QK=QK, KV=KV, R=R, FULL_K=FULL_K,
        BLOCK_S=_BLOCK_S, BLOCK_K=_BLOCK_K, BLOCK_N=_BLOCK_N,
    )
    return q_nope_out


def fused_v_correction(attn, A, B, scaling, base_view, qk_nope, v_head_dim):
    """attn (S,H,KV); A (R,KV); B (H*FULL_K,R); base_view (S,H,V) in-place."""
    S, H, KV = attn.shape
    R = A.shape[0]
    FULL_K = B.shape[0] // H
    grid = (triton.cdiv(S, _BLOCK_S), H)
    _fused_v_kernel[grid](
        attn, A, B, base_view, S, _scaling_tensor(scaling, attn.device),
        attn.stride(0), attn.stride(1), attn.stride(2),
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        base_view.stride(0), base_view.stride(1), base_view.stride(2),
        H=H, KV=KV, V=v_head_dim, R=R, FULL_K=FULL_K, QK_OFF=qk_nope,
        BLOCK_S=_BLOCK_S, BLOCK_K=_BLOCK_K, BLOCK_N=_BLOCK_N,
    )
    return base_view
