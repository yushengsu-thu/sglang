"""Single-GPU micro-bench for the absorbed-MLA kv_b LoRA step kernels vs full-gemm floor.

Real Kimi-K2.5 per-TP-rank shapes (tp8): H=8 heads, qk_nope=v_head_dim=128, kv_lora_rank=512, rank=16.
For each of the 4 step kernels we report:
  * the kernel time (single-LoRA batch_info, num_segments=1)
  * a plain cuBLAS gemm of the SAME (M,K,N)  -> "is the kernel slower than a same-size gemm?"
  * the corresponding FULL base bmm (q_nope@w_kc / attn@w_vc) -> the user's floor:
    each lora gemm (fewer FLOPs) should be <= the full gemm.
Run: CUDA_VISIBLE_DEVICES=0 python3 bench_kv_b_kernels.py
"""
import torch, triton
from triton.testing import do_bench
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.lora.triton_ops import step_a_q_fwd, step_b_q_fwd, step_a_v_fwd, step_b_v_fwd

dev = "cuda"; dt = torch.bfloat16
H, QK, V, KV, R = 8, 128, 128, 512, 16
FULL_K = QK + V
SCALE = 0.3137


def make_bi(S):
    return LoRABatchInfo(
        use_cuda_graph=False, bs=S, num_segments=1,
        seg_indptr=torch.tensor([0, S], dtype=torch.int32, device=dev),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=dev),
        lora_ranks=torch.tensor([R], dtype=torch.int32, device=dev),
        scalings=torch.tensor([SCALE], dtype=torch.float32, device=dev),
        max_len=S, seg_lens=torch.tensor([S], dtype=torch.int32, device=dev),
        permutation=None,
    )


def ms(fn):
    return do_bench(fn, warmup=50, rep=200)


def bench(S):
    bi = make_bi(S)
    q_nope = torch.randn(S, H, QK, device=dev, dtype=dt)
    attn   = torch.randn(S, H, KV, device=dev, dtype=dt)
    A = torch.randn(1, R, KV, device=dev, dtype=dt)            # lora A (rank, kv_lora_rank)
    B = torch.randn(1, H * FULL_K, R, device=dev, dtype=dt)    # lora B (H*FULL_K, rank)
    w_kc = torch.randn(H, QK, KV, device=dev, dtype=dt)        # full K-absorb weight
    w_vc = torch.randn(H, KV, V, device=dev, dtype=dt)         # full V-absorb weight
    q_out = torch.randn(S, H, KV, device=dev, dtype=dt)
    v_out = torch.randn(S, H, V, device=dev, dtype=dt)
    la_q = step_a_q_fwd(q_nope, B, bi, FULL_K)                 # (S,H,R)
    la_v = step_a_v_fwd(attn, A, bi)                           # (S,H,R)

    qT, aT = q_nope.transpose(0, 1).contiguous(), attn.transpose(0, 1).contiguous()
    laqT = la_q.transpose(0, 1).contiguous()
    rows = S * H
    res = {
        # kernel,                          same-size cuBLAS,                                  full base bmm
        "step_a_q (S,H,128)->R": (lambda: step_a_q_fwd(q_nope, B, bi, FULL_K),
                                  lambda: torch.bmm(qT, torch.randn(H, QK, R, device=dev, dtype=dt)),
                                  lambda: torch.bmm(qT, w_kc)),                       # full q_nope@w_kc
        "step_b_q (S,H,R)->512": (lambda: step_b_q_fwd(la_q, A, bi, q_out),
                                  lambda: torch.matmul(la_q.reshape(rows, R), A[0]),
                                  lambda: torch.bmm(qT, w_kc)),
        "step_a_v (S,H,512)->R": (lambda: step_a_v_fwd(attn, A, bi),
                                  lambda: torch.matmul(attn.reshape(rows, KV), A[0].t()),
                                  lambda: torch.bmm(aT, w_vc)),                       # full attn@w_vc
        "step_b_v (S,H,R)->128": (lambda: step_b_v_fwd(la_v, B, bi, v_out, QK, V),
                                  lambda: torch.bmm(laqT, torch.randn(H, R, V, device=dev, dtype=dt)),
                                  lambda: torch.bmm(aT, w_vc)),
    }
    print(f"\n===== S={S} (H={H}, qk={QK}, kv={KV}, rank={R}) =====")
    print(f"{'op':24} {'kernel us':>10} {'cublas-same us':>15} {'full-bmm us':>12}  verdict")
    tot_k = 0.0
    for name, (kfn, cfn, ffn) in res.items():
        k, c, f = ms(kfn) * 1e3, ms(cfn) * 1e3, ms(ffn) * 1e3
        tot_k += k
        v = "OK(<=full)" if k <= f else f"SLOWER than full by {k/f:.1f}x"
        print(f"{name:24} {k:10.1f} {c:15.1f} {f:12.1f}  {v}")
    print(f"{'-- 4 step kernels sum':24} {tot_k:10.1f} us")


if __name__ == "__main__":
    print("torch", torch.__version__, "| dev", torch.cuda.get_device_name(0))
    for S in (16, 32, 64):
        bench(S)
