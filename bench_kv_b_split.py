"""CUDA-graph bench: SPLIT single-LoRA step_a/step_b vs current Triton step kernels vs base-bmm budget.
Key question: is single-LoRA step_a <= the base bmm (so two-stream fully hides it)?
Shapes = Kimi decode bs=64 per TP rank. Run: CUDA_VISIBLE_DEVICES=0 python3 bench_kv_b_split.py
"""
import time, torch
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.lora.triton_ops import step_a_q_fwd, step_b_q_fwd, step_a_v_fwd, step_b_v_fwd
from sglang.srt.lora.triton_ops.kv_b_lora_single_fused import (
    single_a_q_fwd, single_b_q_fwd, single_a_v_fwd, single_b_v_fwd)

dev, dt = "cuda", torch.bfloat16
S, H, QK, V, KV, R, FULL_K, SCALE = 64, 8, 128, 128, 512, 16, 256, 0.3137
SCL = torch.tensor([SCALE], dtype=torch.float32, device=dev)


def bi():
    return LoRABatchInfo(use_cuda_graph=False, bs=S, num_segments=1,
        seg_indptr=torch.tensor([0, S], dtype=torch.int32, device=dev),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=dev),
        lora_ranks=torch.tensor([R], dtype=torch.int32, device=dev),
        scalings=torch.tensor([SCALE], dtype=torch.float32, device=dev),
        max_len=S, seg_lens=torch.tensor([S], dtype=torch.int32, device=dev), permutation=None)


def graph_us(fn, reps=2000):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): g.replay()
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
oq, ov = torch.zeros(S, H, KV, device=dev, dtype=dt), torch.zeros(S, H, V, device=dev, dtype=dt)
fbq, fbv = torch.zeros(H, S, KV, device=dev, dtype=dt), torch.zeros(H, S, V, device=dev, dtype=dt)

# correctness: split single (a+b) vs existing 2-kernel path
oqr = torch.randn(S, H, KV, device=dev, dtype=dt); base_q = oqr.clone()
o1 = oqr.clone(); step_b_q_fwd(step_a_q_fwd(q, B, b, FULL_K), A, b, o1)
o2 = oqr.clone(); single_b_q_fwd(single_a_q_fwd(q, B[0]), A[0], SCL, o2)
ovr = torch.randn(S, H, V, device=dev, dtype=dt)
ov1 = ovr.clone(); step_b_v_fwd(step_a_v_fwd(attn, A, b), B, b, ov1, QK, V)
ov2 = ovr.clone(); single_b_v_fwd(single_a_v_fwd(attn, A[0]), B[0], SCL, ov2, QK, V)
print(f"correctness q maxΔ={(o1-o2).abs().max():.2e}  v maxΔ={(ov1-ov2).abs().max():.2e}")

laq = single_a_q_fwd(q, B[0]); lav = single_a_v_fwd(attn, A[0])
rows = [
    ("base q full-bmm (overlap budget)", lambda: torch.bmm(qT, w_kc, out=fbq)),
    ("base v full-bmm (overlap budget)", lambda: torch.bmm(aT, w_vc, out=fbv)),
    ("step_a_q  CURRENT", lambda: step_a_q_fwd(q, B, b, FULL_K)),
    ("step_a_q  single  ", lambda: single_a_q_fwd(q, B[0])),
    ("step_a_v  CURRENT", lambda: step_a_v_fwd(attn, A, b)),
    ("step_a_v  single  ", lambda: single_a_v_fwd(attn, A[0])),
    ("step_b_q  CURRENT", lambda: step_b_q_fwd(laq.transpose(0,1).contiguous().transpose(0,1) if False else step_a_q_fwd(q,B,b,FULL_K), A, b, oq)),
    ("step_b_q  single  ", lambda: single_b_q_fwd(laq, A[0], SCL, oq)),
    ("step_b_v  single  ", lambda: single_b_v_fwd(lav, B[0], SCL, ov, QK, V)),
]
print(f"{'op':36}{'us (graph)':>12}")
for name, fn in rows:
    print(f"{name:36}{graph_us(fn):12.2f}")
print("\n>>> KEY: step_a_v single <= base v full-bmm  => step_a fully hides under two-stream overlap")
