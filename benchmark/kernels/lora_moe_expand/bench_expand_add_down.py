"""Self-contained micro-benchmark + correctness check for _moe_lora_expand_add_kernel
(LoRA-B expand-add, down-proj GEMM).

Companion to ``bench_shrink_splitk.py`` (which covers the LoRA-A shrink). Scoped to the
qwen3.5-35b local-EP down-proj expand: tp=4/ep=4 -> 64 local experts, N (down output
hidden) = 2048, rank = 16, top_k = 8. The down-proj expand uses the fused
sum-all-reduce + routed-weight variant: each of a token's top_k per-expert deltas is
scaled by its routing weight and atomic-added into the single per-token output row.

P0 scope: bs=64, rank=16 first.

  python3 bench_expand_add_down.py --mode bench
  python3 bench_expand_add_down.py --mode correctness   # sweeps block_m {16,32,64} x bs {16,64}
  python3 bench_expand_add_down.py --mode profile --iters 2   # eager, for ncu
  python3 bench_expand_add_down.py --mode sweep              # bs=64 r=16: block_m x group_m x warps

correctness mode also guards the routing/tiling block-size contract: the launcher must
tile ``expert_ids`` with the same block size the routing buffers were aligned with, else
expert_ids overruns -> IMA (the same class of bug as the shrink f2adddd regression).
"""
from __future__ import annotations

import argparse

import torch
import triton
import triton.testing

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.virtual_experts import (
    _fused_virtual_topk_ids,
    fused_sanitize_expert_ids,
)
from sglang.srt.lora.trtllm_moe.specialized_expand import _invoke_moe_lora_expand_add


def make_inputs(bs, num_experts, top_k, n, rank, dtype, device):
    """Down-proj expand inputs.

    ``intermediate`` is the LoRA-A shrink output [bs*top_k, rank]; ``lora_b`` is the
    per-(lora, expert) down LoRA-B weight [1, num_experts, N, rank].
    """
    torch.manual_seed(0)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(bs)]
    ).to(torch.int32)
    # Positive routing weights (down-proj scales each per-expert delta by these).
    topk_weights = torch.rand(bs, top_k, device=device, dtype=torch.float32) * 0.9 + 0.1
    token_lora_mapping = torch.zeros(bs, device=device, dtype=torch.int32)
    intermediate = torch.randn(bs * top_k, rank, device=device, dtype=dtype) * 0.1
    lora_b = torch.randn(1, num_experts, n, rank, device=device, dtype=dtype) * 0.1
    return topk_ids, topk_weights, token_lora_mapping, intermediate, lora_b


def build_v1_routing(topk_ids, token_lora_mapping, num_experts, block_m):
    """Single-adapter (max_loras=1) virtual-expert routing, tiled at ``block_m``.

    Mirrors ``_get_routing`` in virtual_experts.py: virtual topk ids -> align ->
    tight trim -> sanitize. The trim+sanitize matter because the launcher reads one
    ``expert_ids`` entry per M-block.
    """
    virtual_topk_ids, _, virtual_num_experts = _fused_virtual_topk_ids(
        topk_ids, token_lora_mapping, num_experts, shared_outer=False, max_loras=1
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        virtual_topk_ids, block_m, virtual_num_experts
    )
    num_tokens = topk_ids.numel()
    max_nonempty = min(num_tokens, virtual_num_experts)
    tight = triton.cdiv(num_tokens + max_nonempty * (block_m - 1), block_m) * block_m
    return (
        sorted_token_ids[:tight],
        fused_sanitize_expert_ids(expert_ids[: tight // block_m], virtual_num_experts),
        num_tokens_post_padded,
    )


def expand(
    intermediate,
    lora_b,
    topk_ids,
    topk_weights,
    routing,
    block_m,
    block_n=64,
    group_m=1,
    num_warps=4,
    mul_routed_weight=True,
    fuse_sum_all_reduce=True,
):
    sorted_token_ids, expert_ids, num_tokens_post_padded = routing
    lora_b_virtual = lora_b.reshape(lora_b.shape[0] * lora_b.shape[1], *lora_b.shape[2:])
    n = lora_b.shape[2]
    bs, top_k = topk_ids.shape
    # FUSE_SUM_ALL_REDUCE atomic-adds the top_k deltas into one row -> zero each call.
    # (Without it, each (token,expert) slot is written once -> bs*top_k rows.)
    out_rows = bs if fuse_sum_all_reduce else bs * top_k
    output = torch.zeros(
        out_rows, n, dtype=intermediate.dtype, device=intermediate.device
    )
    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,  # overridden to 128 by the launcher when N % 128 == 0
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
    }
    _invoke_moe_lora_expand_add(
        intermediate,
        lora_b_virtual,
        output,
        # kernel indexes topk_weights flat by virtual-token id in [0, bs*top_k).
        topk_weights.reshape(-1),
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        mul_routed_weight,
        fuse_sum_all_reduce,
    )
    return output


def ref_expand(
    intermediate, lora_b, topk_ids, topk_weights,
    mul_routed_weight=True, fuse_sum_all_reduce=True,
):
    bs, top_k = topk_ids.shape
    n = lora_b.shape[2]
    b = lora_b[0].float()  # [num_experts, N, R]
    inter = intermediate.float()  # [bs*top_k, R]
    rows = bs if fuse_sum_all_reduce else bs * top_k
    out = torch.zeros(rows, n, device=intermediate.device, dtype=torch.float32)
    for m in range(bs):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            vt = m * top_k + k
            delta = inter[vt] @ b[e].t()  # [N]
            if mul_routed_weight:
                delta = delta * float(topk_weights[m, k].item())
            if fuse_sum_all_reduce:
                out[m] += delta
            else:
                out[vt] = delta
    return out


def bench_ms(fn, warmup=25, rep=100, cudagraph=True, inner=200):
    """Per-call milliseconds.

    With ``cudagraph``, capture ``inner`` back-to-back ``fn()`` calls in ONE graph and
    divide the measured replay time by ``inner``. A single fn()-per-graph
    ``do_bench(g.replay)`` floors at ~8-10us for ANY tiny op -- it measures the fixed
    per-replay launch/dispatch overhead, not the kernel. Amortizing over ``inner``
    back-to-back calls drives that overhead to ~0 and exposes the true device time.
    (Same technique as bench_shrink_splitk.py.)
    """
    torch.cuda.synchronize()
    if cudagraph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for _ in range(inner):
                    fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                fn()
        torch.cuda.synchronize()
        ms = triton.testing.do_bench(g.replay, warmup=warmup, rep=rep) / inner
    else:
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    torch.cuda.synchronize()
    return float(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["bench", "correctness", "profile", "sweep"],
        default="bench",
    )
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--num-experts", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n", type=int, default=2048, help="down-proj output hidden")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--group-m", type=int, default=1)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--tol", type=float, default=5e-2)
    args = ap.parse_args()

    dev = "cuda"

    if args.mode == "sweep":
        # P0: bs=64, rank=16. Tunable knobs are BLOCK_SIZE_M / GROUP_SIZE_M / num_warps
        # (the launcher forces BLOCK_SIZE_N=128 for N%128==0 and num_stages=1).
        topk_ids, tkw, tlm, inter, lora_b = make_inputs(
            args.bs, args.num_experts, args.top_k, args.n, args.rank,
            torch.bfloat16, dev,
        )
        best = None
        for block_m in [16, 32, 64]:
            routing = build_v1_routing(topk_ids, tlm, args.num_experts, block_m)
            for group_m in [1, 4, 8]:
                for nw in [2, 4, 8]:
                    f = lambda bm=block_m, gm=group_m, w=nw, r=routing: expand(
                        inter, lora_b, topk_ids, tkw, r, bm, group_m=gm, num_warps=w
                    )
                    try:
                        us = bench_ms(f, warmup=15, rep=60) * 1000
                    except Exception:
                        continue
                    tag = f"block_m={block_m} group_m={group_m} warps={nw}"
                    if best is None or us < best[0]:
                        best = (us, tag)
                    print(f"  {us:7.2f} us  {tag}")
        print(f"\nBEST bs={args.bs} r={args.rank} N={args.n}: {best[0]:.2f} us  {best[1]}")
        return

    topk_ids, tkw, tlm, inter, lora_b = make_inputs(
        args.bs, args.num_experts, args.top_k, args.n, args.rank,
        torch.bfloat16, dev,
    )
    routing = build_v1_routing(topk_ids, tlm, args.num_experts, args.block_m)
    fn = lambda: expand(
        inter, lora_b, topk_ids, tkw, routing,
        args.block_m, args.block_n, args.group_m, args.num_warps,
    )

    if args.mode == "correctness":
        # Guard the routing/tiling block-size contract: _invoke_moe_lora_expand_add tiles
        # expert_ids with config["BLOCK_SIZE_M"] (one entry per M-block); routing must be
        # aligned with the SAME block. Build routing AND config with the same block_m per
        # iteration, but SWEEP block_m so any hardcoded value diverges from the routing at
        # the other block sizes; sweep bs too so the overrun is deterministic enough to
        # fault. Reference accumulates in fp32, kernel accumulates per-expert bf16
        # atomic-adds -> generous abs tol.
        block_ms = sorted({args.block_m, 16, 32, 64})
        batch_sizes = sorted({args.bs, 16, 64})
        failures = 0
        for bs in batch_sizes:
            tk, tkw_b, tlm_b, inter_b, lb = make_inputs(
                bs, args.num_experts, args.top_k, args.n, args.rank,
                torch.bfloat16, dev,
            )
            ref = ref_expand(inter_b, lb, tk, tkw_b)
            for bm in block_ms:
                routing_bm = build_v1_routing(tk, tlm_b, args.num_experts, bm)
                out = expand(
                    inter_b, lb, tk, tkw_b, routing_bm,
                    bm, args.block_n, args.group_m, args.num_warps,
                ).float()
                err = float((out - ref).abs().max().item())
                rel = err / float(ref.abs().max().item() + 1e-9)
                ok = err <= args.tol
                failures += int(not ok)
                print(
                    f"{'PASS' if ok else 'FAIL'} bs={bs:<3d} block_m={bm:<2d} "
                    f"max_abs_err={err:.4e} rel={rel:.2e}"
                )
        if failures:
            raise SystemExit(1)
    elif args.mode == "profile":
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
    else:
        ms = bench_ms(fn)
        print(
            f"BENCH expand_add down bs={args.bs} r={args.rank} N={args.n} top_k={args.top_k} "
            f"block_m={args.block_m} group_m={args.group_m} warps={args.num_warps}: "
            f"{ms * 1000:.2f} us (amortized true device time)"
        )


if __name__ == "__main__":
    main()
