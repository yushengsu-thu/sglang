"""Bench + correctness: single-LoRA FUSED kv_b kernels vs the 2-kernel step path vs full-gemm floor.
Real Kimi per-rank shapes (H=8, qk=v=128, kv=512, rank=16). Run: CUDA_VISIBLE_DEVICES=0 python3 ...
"""
import torch, triton
from triton.testing import do_bench
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.lora.triton_ops import step_a_q_fwd, step_b_q_fwd, step_a_v_fwd, step_b_v_fwd
from sglang.srt.lora.triton_ops.kv_b_lora_single_fused import fused_q_correction, fused_v_correction

dev, dt = "cuda", torch.bfloat16
H, QK, V, KV, R, FULL_K, SCALE = 8, 128, 128, 512, 16, 256, 0.3137
ms = lambda f: do_bench(f, warmup=50, rep=200) * 1e3


def bi(S):
    return LoRABatchInfo(use_cuda_graph=False, bs=S, num_segments=1,
        seg_indptr=torch.tensor([0, S], dtype=torch.int32, device=dev),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=dev),
        lora_ranks=torch.tensor([R], dtype=torch.int32, device=dev),
        scalings=torch.tensor([SCALE], dtype=torch.float32, device=dev),
        max_len=S, seg_lens=torch.tensor([S], dtype=torch.int32, device=dev), permutation=None)


def run(S):
    b = bi(S)
    q = torch.randn(S, H, QK, device=dev, dtype=dt)
    attn = torch.randn(S, H, KV, device=dev, dtype=dt)
    A = torch.randn(1, R, KV, device=dev, dtype=dt) * 0.1
    B = torch.randn(1, H * FULL_K, R, device=dev, dtype=dt) * 0.1
    w_kc = torch.randn(H, QK, KV, device=dev, dtype=dt)
    w_vc = torch.randn(H, KV, V, device=dev, dtype=dt)
    qT, aT = q.transpose(0, 1).contiguous(), attn.transpose(0, 1).contiguous()
    base_q = torch.randn(S, H, KV, device=dev, dtype=dt)
    base_v = torch.randn(S, H, V, device=dev, dtype=dt)

    # correctness: 2-kernel path vs fused
    o_ref = base_q.clone(); step_b_q_fwd(step_a_q_fwd(q, B, b, FULL_K), A, b, o_ref)
    o_fus = base_q.clone(); fused_q_correction(q, B[0], A[0], SCALE, o_fus)
    eq = (o_ref - o_fus).abs().max().item()
    ov_ref = base_v.clone(); step_b_v_fwd(step_a_v_fwd(attn, A, b), B, b, ov_ref, QK, V)
    ov_fus = base_v.clone(); fused_v_correction(attn, A[0], B[0], SCALE, ov_fus, QK, V)
    ev = (ov_ref - ov_fus).abs().max().item()

    # speed — accumulate into a fixed preallocated buffer (no per-iter clone; values drift, timing is clean)
    oq, ov = base_q, base_v
    t_q2 = ms(lambda: step_b_q_fwd(step_a_q_fwd(q, B, b, FULL_K), A, b, oq))
    t_qf = ms(lambda: fused_q_correction(q, B[0], A[0], SCALE, oq))
    t_qfull = ms(lambda: torch.bmm(qT, w_kc))
    t_v2 = ms(lambda: step_b_v_fwd(step_a_v_fwd(attn, A, b), B, b, ov, QK, V))
    t_vf = ms(lambda: fused_v_correction(attn, A[0], B[0], SCALE, ov, QK, V))
    t_vfull = ms(lambda: torch.bmm(aT, w_vc))

    print(f"\n===== S={S} =====   (q correctness maxΔ={eq:.2e}, v correctness maxΔ={ev:.2e})")
    print(f"{'':14}{'2-kernel us':>12}{'FUSED us':>10}{'full-bmm us':>12}{'  speedup vs 2-kernel':>22}")
    print(f"{'q-correction':14}{t_q2:12.1f}{t_qf:10.1f}{t_qfull:12.1f}{t_q2/t_qf:21.2f}x")
    print(f"{'v-correction':14}{t_v2:12.1f}{t_vf:10.1f}{t_vfull:12.1f}{t_v2/t_vf:21.2f}x")
    print(f"{'TOTAL':14}{t_q2+t_v2:12.1f}{t_qf+t_vf:10.1f}{t_qfull+t_vfull:12.1f}")


if __name__ == "__main__":
    print("dev", torch.cuda.get_device_name(0))
    for S in (16, 32, 64):
        run(S)
