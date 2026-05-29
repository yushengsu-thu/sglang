"""Bench the two MoE-LoRA kernels at the REAL shapes captured on a live server.

Captured for Qwen3.5-35B-A3B-FP8 (tp4 ep4, max_loras=1, rank16, bf16); see
results/qwen35_shapes.md. Two stages per MoE layer, all 256 experts per rank:

  gate_up: shrink hidden(T,2048) w(256,32,2048) -> (T*8,32);  expand w(256,1024,16) -> (T,8,1024)   topk=8
  down   : shrink hidden(T*8,512) w(256,16,512) -> (T*8,16);  expand w(256,2048,16) -> (T,2048)      topk=1 (shrink), sum-reduce

Runs do_bench for each kernel at T in {decode bs 16/32/64, a prefill chunk}.

Usage: python bench_real_shapes.py [--model qwen35] [--ts 16,32,64,2048]
"""
import argparse

import torch
import triton

from sglang.srt.lora.triton_ops.virtual_experts import (
    _get_moe_lora_shrink_split_k,
    _invoke_moe_lora_expand_add,
    _invoke_moe_lora_shrink_splitk,
)
# reuse the validated routing/config mirror from the testbed
from testbed import build_routing, stage_config

DEV = "cuda"
DT = torch.bfloat16

# Real per-rank shapes captured live (see results/SHAPES_FOR_CHUNAN.md). kimi is derived.
# (name, E, shrink_K, shrink_rankout, expand_N, expand_R, topk, mul_routed, sum_reduce)
MODEL_STAGES = {
    "qwen35": [  # tp4 ep4, E=256, H=2048, M=1024  [server-captured]
        ("gate_up", 256, 2048, 32, 1024, 16, 8, False, False),
        ("down",    256, 512, 16, 2048, 16, 1, True, True),
    ],
    "qwen3vl": [  # tp4 ep4, E=128, H=2048, M=1536  [server-captured]
        ("gate_up", 128, 2048, 32, 1536, 16, 8, False, False),
        ("down",    128, 768, 16, 2048, 16, 1, True, True),
    ],
    "kimi": [  # tp8 no-EP, E=384, H=7168, M=256  [DERIVED from config; verify on capture]
        ("gate_up", 384, 7168, 32, 256, 16, 8, False, False),
        ("down",    384, 256, 16, 7168, 16, 1, True, True),
    ],
}


def make_topk(T, topk, E, g):
    ids = torch.empty(T, topk, dtype=torch.int32, device=DEV)
    for i in range(T):
        ids[i] = torch.randperm(E, generator=g, device=DEV)[:topk].to(torch.int32)
    w = torch.rand(T, topk, dtype=torch.float32, device=DEV, generator=g)
    return ids, w / w.sum(1, keepdim=True)


def bench_stage(name, E, K, rank_out, N, R, topk, mul_routed, sum_reduce, T):
    g = torch.Generator(device=DEV).manual_seed(0)
    tlm = torch.zeros(T, dtype=torch.int32, device=DEV)  # single lora active
    topk_ids, topk_w = make_topk(T, topk, E, g)

    # --- shrink ---
    hidden = torch.randn(T, K, dtype=DT, device=DEV, generator=g)
    lora_a = torch.randn(E, rank_out, K, dtype=DT, device=DEV, generator=g) * 0.02
    a_cfg = stage_config(lora_a, topk, DT, T)
    s_ids, e_ids, ntpp, _ = build_routing(topk_ids, tlm, E, a_cfg["BLOCK_SIZE_M"], 1)
    interm = torch.zeros(T * topk, rank_out, dtype=DT, device=DEV)
    split_k = _get_moe_lora_shrink_split_k(lora_a, s_ids, a_cfg)

    def _shrink():
        _invoke_moe_lora_shrink_splitk(hidden, lora_a, interm, topk_ids, s_ids, e_ids, ntpp, topk, a_cfg)

    # --- expand --- (reads R cols from the rank_out-wide intermediate)
    lora_b = torch.randn(E, N, R, dtype=DT, device=DEV, generator=g) * 0.02
    b_cfg = stage_config(lora_b, 1, DT, T)
    s_ids_b, e_ids_b, ntpp_b, _ = build_routing(topk_ids, tlm, E, b_cfg["BLOCK_SIZE_M"], 1)
    interm_e = interm[:, :R].contiguous()
    out = torch.zeros((T, N) if sum_reduce else (T * topk, N), dtype=DT, device=DEV)

    def _expand():
        _invoke_moe_lora_expand_add(interm_e, lora_b, out, topk_w, topk_ids, s_ids_b, e_ids_b, ntpp_b, b_cfg, mul_routed, sum_reduce)

    t_s = triton.testing.do_bench(_shrink, warmup=50, rep=200)
    t_e = triton.testing.do_bench(_expand, warmup=50, rep=200)
    print(f"  {name:8s} T={T:<6d} shrink={t_s*1e3:7.2f}us (split_k={split_k}, blk={a_cfg['BLOCK_SIZE_M']},{a_cfg.get('BLOCK_SIZE_N')},{a_cfg['BLOCK_SIZE_K']})  expand={t_e*1e3:7.2f}us")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_STAGES), default="qwen35")
    p.add_argument("--ts", default="16,32,64,2048")
    args = p.parse_args()
    Ts = [int(x) for x in args.ts.split(",")]
    stages = MODEL_STAGES[args.model]
    print(f"== {args.model} real-shape MoE-LoRA kernel latency (per TP rank) ==")
    for T in Ts:
        regime = "decode" if T <= 256 else "prefill"
        print(f"-- T={T} ({regime}) --")
        for stage in stages:
            bench_stage(*stage, T=T)


if __name__ == "__main__":
    main()
