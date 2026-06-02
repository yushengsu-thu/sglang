"""CUDA-GRAPH bench to MATCH real serving (decode is graph-captured; launch overhead ~0).

Goal first: reproduce the user's profile numbers (base v full-gemm ~3.5us, lora-v non-overlap ~15us)
so the testbed is trustworthy, THEN measure the fused kernel in the SAME graph regime.
Shapes = real Kimi decode bs=64 per TP rank: S=64, H=8, qk=v=128, kv=512, rank=16.
Run: CUDA_VISIBLE_DEVICES=0 python3 bench_kv_b_graph.py
"""
import time, torch
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.lora.triton_ops import step_a_q_fwd, step_b_q_fwd, step_a_v_fwd, step_b_v_fwd
from sglang.srt.lora.triton_ops.kv_b_lora_single_fused import fused_q_correction, fused_v_correction

dev, dt = "cuda", torch.bfloat16
S, H, QK, V, KV, R, FULL_K, SCALE = 64, 8, 128, 128, 512, 16, 256, 0.3137


def bi():
    return LoRABatchInfo(use_cuda_graph=False, bs=S, num_segments=1,
        seg_indptr=torch.tensor([0, S], dtype=torch.int32, device=dev),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=dev),
        lora_ranks=torch.tensor([R], dtype=torch.int32, device=dev),
        scalings=torch.tensor([SCALE], dtype=torch.float32, device=dev),
        max_len=S, seg_lens=torch.tensor([S], dtype=torch.int32, device=dev), permutation=None)


def graph_us(fn, reps=2000):
    """Capture fn() in a CUDA graph and time replay (launch overhead amortized, like serving)."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6


b = bi()
q = torch.randn(S, H, QK, device=dev, dtype=dt)
attn = torch.randn(S, H, KV, device=dev, dtype=dt)
A = torch.randn(1, R, KV, device=dev, dtype=dt) * 0.1
B = torch.randn(1, H * FULL_K, R, device=dev, dtype=dt) * 0.1
w_kc = torch.randn(H, QK, KV, device=dev, dtype=dt)
w_vc = torch.randn(H, KV, V, device=dev, dtype=dt)
qT, aT = q.transpose(0, 1).contiguous(), attn.transpose(0, 1).contiguous()
out_q = torch.zeros(S, H, KV, device=dev, dtype=dt)
out_v = torch.zeros(S, H, V, device=dev, dtype=dt)
fb_q = torch.zeros(H, S, KV, device=dev, dtype=dt)
fb_v = torch.zeros(H, S, V, device=dev, dtype=dt)


def lora_q_2k():
    step_b_q_fwd(step_a_q_fwd(q, B, b, FULL_K), A, b, out_q)
def lora_v_2k():
    step_b_v_fwd(step_a_v_fwd(attn, A, b), B, b, out_v, QK, V)


print("dev", torch.cuda.get_device_name(0), "| CUDA-GRAPH regime (matches serving) | S=64")
rows = [
    ("base q full-bmm (q_nope@w_kc)", lambda: torch.bmm(qT, w_kc, out=fb_q)),
    ("base v full-bmm (attn@w_vc)",   lambda: torch.bmm(aT, w_vc, out=fb_v)),
    ("lora q  2-kernel (a+b)",        lora_q_2k),
    ("lora v  2-kernel (a+b)",        lora_v_2k),
    ("lora q  FUSED",                 lambda: fused_q_correction(q, B[0], A[0], SCALE, out_q)),
    ("lora v  FUSED",                 lambda: fused_v_correction(attn, A[0], B[0], SCALE, out_v, QK, V)),
]
print(f"{'op':34}{'us (graph replay)':>18}")
for name, fn in rows:
    print(f"{name:34}{graph_us(fn):18.2f}")
print("\n>>> check: does 'base v full-bmm' ~= 3.5us and 'lora v 2-kernel' ~= 15us (your profile)?")
