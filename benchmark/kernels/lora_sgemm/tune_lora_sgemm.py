"""
Auto-tuning script for the dense-attention LoRA sgemm A/B Triton kernels.

These are the kernels the `triton` LoRA backend uses for attention/dense projections:
  - sgemm_lora_a   (shrink, A): x @ A^T, input_dim -> N (= num_slices * rank)
  - sgemm_lora_b   (expand, B, S=1): o_proj / down_proj
  - gate_up_lora_b (expand, B, S=2)
  - qkv_lora_b     (expand, B, S>=3)

They historically used hardcoded block sizes and never set num_warps/num_stages.
This script sweeps {BLOCK_S, BLOCK_N, BLOCK_K, num_warps, num_stages} per
(kernel, K, R, S, max_len) and writes the best config as JSON keyed by max_len,
mirroring benchmark/kernels/lora_csgmv/tune_lora_csgmv.py. The server picks the
files up at runtime via lora_tuning_config.get_sgemm_a/b_config.

IMPORTANT — tune the EXACT shapes the model requests at runtime. The config key
(K, R, S) is exactly what the launcher passes to get_sgemm_{a,b}_config:
  sgemm_a:  K = input_dim,           R = N = num_slices*rank (kernel tiles N), S = num_slices
  sgemm_b:  K = per-slice out width, R = adapter rank (reduction dim),          S = num_slices
Read them off a live server log:
  grep -oE "sgemm_[ab] \\(K=[0-9]+, R=[0-9]+, S=[0-9]+" /tmp/server.log | sort -u
then pass them via --shapes "sgemm_a:2048:48:3,sgemm_b:2048:16:3,...".

Usage:
    python3 benchmark/kernels/lora_sgemm/tune_lora_sgemm.py \
        --shapes "sgemm_a:2048:48:3,sgemm_a:2048:64:4,sgemm_b:2048:16:3,sgemm_b:7168:32:1" \
        --decode-batch-sizes 1 2 3 4 5 6 7 8 16 24 32 48 64 \
        --prefill-num-tokens 512 1024 2048 4096 8192 --seed 0
"""

import argparse
import json
import os
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import triton

from sglang.srt.lora.triton_ops import (
    gate_up_lora_b_fwd,
    qkv_lora_b_fwd,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)
from sglang.srt.lora.triton_ops.lora_tuning_config import (
    get_sgemm_config_file_name,
    set_sgemm_config_override,
)
from sglang.srt.lora.utils import LoRABatchInfo

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def build_batch_info(max_len: int, rank: int, device: torch.device) -> LoRABatchInfo:
    """Single-adapter batch, one segment of length max_len (slot 1). `rank` goes
    into lora_ranks so the kernel's N=min(N, rank*S) / K=min(K, rank) clips match."""
    return LoRABatchInfo(
        use_cuda_graph=False, bs=1, num_segments=1, max_len=max_len,
        seg_indptr=torch.tensor([0, max_len], dtype=torch.int32, device=device),
        weight_indices=torch.ones(1, dtype=torch.int32, device=device),
        lora_ranks=torch.tensor([0, rank], dtype=torch.int32, device=device),
        scalings=torch.ones(2, dtype=torch.float32, device=device),
        seg_lens=torch.tensor([max_len], dtype=torch.int32, device=device),
        permutation=None,
    )


def timed_cuda_ms(fn, warmup=5, trials=15) -> float:
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def _bench(fn) -> Optional[float]:
    try:
        fn(); torch.cuda.synchronize()
    except Exception:
        return None
    return timed_cuda_ms(fn)


# ---- search spaces ----

def shrink_space(N: int) -> List[Dict[str, Any]]:
    cfgs = []
    for bs in [16, 32]:
        for bn in [16, 32, 64]:
            if bn > max(16, N): continue
            for bk in [64, 128, 256]:
                for w in [4, 8]:
                    for st in [2, 3, 4]:
                        cfgs.append({"BLOCK_S": bs, "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": w, "num_stages": st})
    return cfgs


def expand_space(R: int) -> List[Dict[str, Any]]:
    cfgs = []
    for bs in [16, 32]:
        for bn in [64, 128, 256]:
            for bk in [16, 32, 64]:
                if bk > max(16, R): continue
                for w in [4, 8]:
                    for st in [2, 3, 4]:
                        cfgs.append({"BLOCK_S": bs, "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": w, "num_stages": st})
    return cfgs


# ---- per-kernel benchmark (drives the real *_fwd via config override) ----

def bench_shrink(cfg, K, N, S, bi, dev, dt):
    M = bi.max_len
    x = torch.randn(M, K, device=dev, dtype=dt)
    w = torch.randn(2, N, K, device=dev, dtype=dt)
    set_sgemm_config_override(cfg)
    try:
        return _bench(lambda: sgemm_lora_a_fwd(x, w, bi, stack_num=S))
    finally:
        set_sgemm_config_override(None)


def bench_expand(cfg, Kout, R, S, bi, dev, dt):
    M = bi.max_len
    set_sgemm_config_override(cfg)
    try:
        if S == 1:
            x = torch.randn(M, R, device=dev, dtype=dt)
            w = torch.randn(2, Kout, R, device=dev, dtype=dt)
            o = torch.zeros(M, Kout, device=dev, dtype=dt)
            return _bench(lambda: sgemm_lora_b_fwd(x, w, bi, o))
        if S == 2:
            x = torch.randn(M, 2 * R, device=dev, dtype=dt)
            w = torch.randn(2, 2 * Kout, R, device=dev, dtype=dt)
            o = torch.zeros(M, 2 * Kout, device=dev, dtype=dt)
            return _bench(lambda: gate_up_lora_b_fwd(x, w, bi, Kout, o))
        # S >= 3: qkv kernel with n_slices=S, each slice width Kout
        x = torch.randn(M, S * R, device=dev, dtype=dt)
        w = torch.randn(2, S * Kout, R, device=dev, dtype=dt)
        o = torch.zeros(M, S * Kout, device=dev, dtype=dt)
        off = torch.tensor([i * Kout for i in range(S + 1)], dtype=torch.int32, device=dev)
        return _bench(lambda: qkv_lora_b_fwd(x, w, bi, off, Kout, o, n_slices=S))
    finally:
        set_sgemm_config_override(None)


# ---- verify (correctness gate): tuned-config output vs default + torch ref ----

def _shrink_out(cfg, K, N, S, bi, dev, dt, x, w):
    set_sgemm_config_override(cfg)
    try:
        return sgemm_lora_a_fwd(x, w, bi, stack_num=S)
    finally:
        set_sgemm_config_override(None)


def _expand_out(cfg, Kout, R, S, bi, dev, dt, x, w):
    set_sgemm_config_override(cfg)
    try:
        if S == 1:
            o = torch.zeros(bi.max_len, Kout, device=dev, dtype=dt)
            return sgemm_lora_b_fwd(x, w, bi, o)
        if S == 2:
            o = torch.zeros(bi.max_len, 2 * Kout, device=dev, dtype=dt)
            return gate_up_lora_b_fwd(x, w, bi, Kout, o)
        o = torch.zeros(bi.max_len, S * Kout, device=dev, dtype=dt)
        off = torch.tensor([i * Kout for i in range(S + 1)], dtype=torch.int32, device=dev)
        return qkv_lora_b_fwd(x, w, bi, off, Kout, o, n_slices=S)
    finally:
        set_sgemm_config_override(None)


def _torch_ref(kernel, Kout, R, S, x, w):
    """Reference output (fp32 matmul) for the synthetic single-adapter layout.
    Adapter weights live in slot 1 (w[1]); rank=R unclipped."""
    A = w[1].float()  # (N,K) shrink  or (slice*Kout, R) expand
    xf = x.float()
    if kernel == "sgemm_a":
        return (xf @ A.t())               # (M, N=R)
    if S == 1:
        return (xf @ A.t())               # (M, Kout)
    # multi-slice expand: slice i uses x[:, i*R:(i+1)*R] @ w[1, i*Kout:(i+1)*Kout].T
    outs = [xf[:, i * R:(i + 1) * R] @ A[i * Kout:(i + 1) * Kout].t() for i in range(S)]
    return torch.cat(outs, dim=1)         # (M, S*Kout)


def verify_shape(kernel, K, R, S, dev, dt, ml=32):
    """Run default + an alt (different BLOCK_K/S to stress reduction order) +
    torch ref; report max-abs-err. PASS if both within bf16 tol."""
    torch.manual_seed(0)
    bi = build_batch_info(ml, R, dev)
    if kernel == "sgemm_a":
        x = torch.randn(ml, K, device=dev, dtype=dt); w = torch.randn(2, R, K, device=dev, dtype=dt)
        run = lambda c: _shrink_out(c, K, R, S, bi, dev, dt, x, w)
        dflt = {"BLOCK_S": 16, "BLOCK_N": 16, "BLOCK_K": 256}
        alt = {"BLOCK_S": 32, "BLOCK_N": 16, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3}
    else:
        nx = (R if S == 1 else S * R); nw = (K if S == 1 else S * K)
        x = torch.randn(ml, nx, device=dev, dtype=dt); w = torch.randn(2, nw, R, device=dev, dtype=dt)
        run = lambda c: _expand_out(c, K, R, S, bi, dev, dt, x, w)
        dflt = {"BLOCK_S": 16, "BLOCK_N": (256 if S == 1 else 64), "BLOCK_K": 16}
        alt = {"BLOCK_S": 32, "BLOCK_N": 128, "BLOCK_K": min(64, max(16, R)), "num_warps": 8, "num_stages": 3}
    out_d = run(dflt).float(); out_a = run(alt).float(); ref = _torch_ref(kernel, K, R, S, x, w)
    e_da = (out_a - out_d).abs().max().item()
    scale = ref.abs().max().item() + 1e-6
    e_dr = (out_d - ref).abs().max().item() / scale
    e_ar = (out_a - ref).abs().max().item() / scale
    ok = (e_dr < 2e-2) and (e_ar < 2e-2) and (e_da / scale < 2e-2)
    print(f"  {kernel} K={K} R={R} S={S}: |alt-def|={e_da:.2e} relerr def={e_dr:.2e} alt={e_ar:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def sort_cfg(c):
    return {k: c[k] for k in ["BLOCK_S", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"] if k in c}


# Historical hardcoded block sizes (the launcher's fallback). Emitted verbatim
# for a bucket when tuning doesn't clearly beat them — block-only (no num_warps/
# num_stages) so the launch is byte-identical to the untuned default => that
# bucket's e2e == base, guaranteeing no regression.
def default_cfg(kernel, S):
    if kernel == "sgemm_a":
        return {"BLOCK_S": 16, "BLOCK_N": 16, "BLOCK_K": 256}
    if S == 1:  # sgemm_lora_b
        return {"BLOCK_S": 16, "BLOCK_N": 256, "BLOCK_K": 16}
    return {"BLOCK_S": 16, "BLOCK_N": 64, "BLOCK_K": 16}  # gate_up (S=2) / qkv (S>=3)


# Minimum microbench speedup over default required to adopt a tuned config.
GUARD_MARGIN = 0.03


def save_config(configs, kernel, K, R, S) -> str:
    fn = get_sgemm_config_file_name(kernel, K, R, S)
    vdir = f"triton_{triton.__version__.replace('.', '_')}"
    d = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..", "..", "..", "python", "sglang", "srt", "lora", "triton_ops", "sgemm_configs", vdir))
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, fn)
    with open(p, "w") as f:
        json.dump(configs, f, indent=4); f.write("\n")
    return p


def tune_shape(kernel, K, R, S, max_lens, dev, dt):
    print(f"\n{'='*70}\nTuning {kernel} K={K} R={R} S={S}\n{'='*70}")
    if kernel == "sgemm_a":
        search = shrink_space(R); rank_for_bi = R          # N=R; rank>=N so no clip
    else:
        search = expand_space(R); rank_for_bi = R          # reduction dim = adapter rank R
    print(f"search={len(search)} configs")
    dflt = default_cfg(kernel, S)
    measure = (lambda c, bi: bench_shrink(c, K, R, S, bi, dev, dt)) if kernel == "sgemm_a" \
        else (lambda c, bi: bench_expand(c, K, R, S, bi, dev, dt))
    best = {}
    for ml in max_lens:
        bi = build_batch_info(ml, rank_for_bi, dev)
        dt_def = measure(dflt, bi)                     # default's time for this bucket
        bc, bt = None, float("inf")
        for c in search:
            t = measure(c, bi)
            if t is not None and t < bt:
                bt, bc = t, c
        if bc is None:
            print(f"  max_len={ml}: NO valid config"); continue
        # no-regression guard: adopt tuned only if it beats default by >= margin;
        # else emit the default verbatim so this bucket stays == base.
        if dt_def is not None and bt >= dt_def * (1.0 - GUARD_MARGIN):
            best[ml] = dict(dflt)
            print(f"  max_len={ml}: keep DEFAULT (tuned {bt:.4f} vs default {dt_def:.4f}ms)")
        else:
            best[ml] = sort_cfg(bc)
            spd = (dt_def / bt) if (dt_def and bt) else float('nan')
            print(f"  max_len={ml}: TUNED {bt:.4f}ms ({spd:.2f}x vs default {dt_def:.4f}) {best[ml]}")
    return best


def main(args):
    if args.seed is not None: torch.manual_seed(args.seed)
    dev = torch.device("cuda:0"); dt = DTYPES[args.dtype]
    max_lens = list(dict.fromkeys(args.decode_batch_sizes + args.prefill_num_tokens))
    shapes = []
    for tok in args.shapes.split(","):
        tok = tok.strip()
        if not tok: continue
        kernel, K, R, S = tok.split(":")
        shapes.append((kernel, int(K), int(R), int(S)))
    if args.verify:
        print(f"VERIFY {len(shapes)} shapes (tuned/alt vs default vs torch ref, bf16 tol)")
        allok = all(verify_shape(kernel, K, R, S, dev, dt) for kernel, K, R, S in shapes)
        print("\nVERIFY", "PASS" if allok else "FAIL")
        import sys; sys.exit(0 if allok else 1)
    print(f"Tuning {len(shapes)} shapes over max_lens={max_lens} dtype={args.dtype}")
    for kernel, K, R, S in shapes:
        best = tune_shape(kernel, K, R, S, max_lens, dev, dt)
        if best:
            print("  saved:", save_config(best, kernel, K, R, S))
    print(f"\nDone at {datetime.now().ctime()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Auto-tune dense-attention LoRA sgemm A/B kernels (explicit shapes)")
    p.add_argument("--shapes", required=True,
                   help='comma list of kernel:K:R:S, e.g. "sgemm_a:2048:48:3,sgemm_b:7168:32:1"')
    p.add_argument("--decode-batch-sizes", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 48, 64])
    p.add_argument("--prefill-num-tokens", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verify", action="store_true",
                   help="correctness gate only: tuned/alt vs default vs torch ref (no tuning)")
    main(p.parse_args())
