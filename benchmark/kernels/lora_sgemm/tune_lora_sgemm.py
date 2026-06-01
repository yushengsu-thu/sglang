"""
Auto-tuning script for the dense-attention LoRA sgemm A/B Triton kernels.

These are the kernels used by the `triton` LoRA backend for attention projections
(and dense MLP when present):
  - sgemm_lora_a   (shrink, A): x @ A^T, input_dim -> num_slices * rank
  - sgemm_lora_b   (expand, B, num_slices=1): o_proj / down_proj
  - qkv_lora_b     (expand, B, num_slices=3)
  - gate_up_lora_b (expand, B, num_slices=2)

They historically used hardcoded block sizes and never set num_warps/num_stages.
This script sweeps {BLOCK_S, BLOCK_N, BLOCK_K, num_warps, num_stages} per
(kernel, K, R, S, max_len) and writes the best config as JSON keyed by max_len,
mirroring benchmark/kernels/lora_csgmv/tune_lora_csgmv.py. The server picks the
files up at runtime via lora_tuning_config.get_sgemm_a/b_config.

Dimensions / terminology:
  K        shrink: input_dim (the large reduction dim, e.g. hidden_size).
           expand: per-slice output width that BLOCK_N tiles (o_proj/down: output_dim;
                   qkv: max(q_dim, kv_dim); gate_up: per-slice output_dim).
  R        max LoRA rank (16 / 32 / 64).
  S        num_slices: qkv=3, gate_up=2, o_proj/down=1 (== sgemm_a stack_num).
  max_len  The M / segment-length regime. For single-adapter decode this is the
           batch size; for prefill it is the number of tokens. Configs are keyed
           by it (closest match at runtime), like csgmv's chunk_size.

The tuner drives the real *_fwd launchers with each candidate config forced via
lora_tuning_config.set_sgemm_config_override, so the benchmarked launch is
byte-for-byte the production launch.

Usage:
    # Qwen3.5-35B attention dims (hidden 2048, kv/o-proj width 512), ranks 16/32/64
    python3 benchmark/kernels/lora_sgemm/tune_lora_sgemm.py \
        --hidden-sizes 2048 512 --ranks 16 32 64 \
        --decode-batch-sizes 1 2 3 4 5 6 7 8 16 24 32 48 64 \
        --prefill-num-tokens 512 1024 2048 4096 8192 --seed 0

    # Kimi-K2.5 (hidden 7168, 2048), ranks 16/32/64
    python3 benchmark/kernels/lora_sgemm/tune_lora_sgemm.py \
        --hidden-sizes 7168 2048 --ranks 16 32 64 \
        --decode-batch-sizes 1 2 3 4 5 6 7 8 16 24 32 48 64 \
        --prefill-num-tokens 512 1024 2048 4096 8192 --seed 0

    # Or derive attention dims from a HF model
    python3 benchmark/kernels/lora_sgemm/tune_lora_sgemm.py \
        --model Qwen/Qwen3-0.6B --ranks 16 32 64
"""

import argparse
import json
import os
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
    """Single-adapter batch with one segment of length max_len (slot 1)."""
    seg_lens = torch.tensor([max_len], dtype=torch.int32, device=device)
    seg_indptr = torch.tensor([0, max_len], dtype=torch.int32, device=device)
    weight_indices = torch.ones(1, dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([0, rank], dtype=torch.int32, device=device)
    scalings = torch.ones(2, dtype=torch.float32, device=device)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        max_len=max_len,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        seg_lens=seg_lens,
        permutation=None,
    )


def timed_cuda_ms(fn, warmup: int = 10, trials: int = 50) -> float:
    """Median GPU time (ms) over `trials`, using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------


def get_shrink_search_space(rank: int, num_slices: int) -> List[Dict[str, Any]]:
    """Candidate configs for sgemm_a (shrink). N = num_slices * rank is small;
    K (input_dim) is large, so BLOCK_K matters most. Trimmed to keep tuning
    tractable (the BLOCK_S=16 M-tile dominates for the small decode regimes)."""
    n = num_slices * rank
    configs = []
    for block_s in [16, 32]:
        for block_n in [16, 32]:
            if block_n > max(16, n):
                continue
            for block_k in [64, 128, 256]:
                for num_warps in [4, 8]:
                    for num_stages in [2, 3, 4]:
                        configs.append(
                            {
                                "BLOCK_S": block_s,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }
                        )
    return configs


def get_expand_search_space(rank: int) -> List[Dict[str, Any]]:
    """Candidate configs for the expand kernels. K (= rank) is the small reduction
    dim; N (output width) is large, so BLOCK_N matters most. BLOCK_K capped at
    rank. Trimmed to keep tuning tractable."""
    configs = []
    for block_s in [16, 32]:
        for block_n in [64, 128, 256]:
            for block_k in [16, 32, 64]:
                if block_k > max(16, rank):
                    continue
                for num_warps in [4, 8]:
                    for num_stages in [2, 3, 4]:
                        configs.append(
                            {
                                "BLOCK_S": block_s,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }
                        )
    return configs


# ---------------------------------------------------------------------------
# Per-kernel benchmark drivers (drive the real *_fwd via config override)
# ---------------------------------------------------------------------------


def _bench(fn) -> Optional[float]:
    try:
        fn()
        torch.cuda.synchronize()
    except Exception:
        return None
    return timed_cuda_ms(fn, warmup=5, trials=15)


def benchmark_shrink(
    config, K, rank, num_slices, batch_info, device, dtype
) -> Optional[float]:
    M = batch_info.max_len
    n = num_slices * rank
    x = torch.randn(M, K, device=device, dtype=dtype)
    weights = torch.randn(2, n, K, device=device, dtype=dtype)
    set_sgemm_config_override(config)
    try:
        return _bench(
            lambda: sgemm_lora_a_fwd(x, weights, batch_info, stack_num=num_slices)
        )
    finally:
        set_sgemm_config_override(None)


def benchmark_expand(
    config, K, rank, num_slices, batch_info, device, dtype
) -> Optional[float]:
    """K is the per-slice output width that BLOCK_N tiles."""
    M = batch_info.max_len
    set_sgemm_config_override(config)
    try:
        if num_slices == 1:
            x = torch.randn(M, rank, device=device, dtype=dtype)
            weights = torch.randn(2, K, rank, device=device, dtype=dtype)
            out = torch.zeros(M, K, device=device, dtype=dtype)
            fn = lambda: sgemm_lora_b_fwd(x, weights, batch_info, out)
        elif num_slices == 2:
            x = torch.randn(M, 2 * rank, device=device, dtype=dtype)
            weights = torch.randn(2, 2 * K, rank, device=device, dtype=dtype)
            out = torch.zeros(M, 2 * K, device=device, dtype=dtype)
            fn = lambda: gate_up_lora_b_fwd(x, weights, batch_info, K, out)
        elif num_slices == 3:
            # Representative qkv: q_dim = kv_dim = K so max_qkv_out_dim = K.
            x = torch.randn(M, 3 * rank, device=device, dtype=dtype)
            weights = torch.randn(2, 3 * K, rank, device=device, dtype=dtype)
            out = torch.zeros(M, 3 * K, device=device, dtype=dtype)
            output_offset = torch.tensor(
                [0, K, 2 * K, 3 * K], dtype=torch.int32, device=device
            )
            fn = lambda: qkv_lora_b_fwd(
                x, weights, batch_info, output_offset, K, out, n_slices=3
            )
        else:
            raise ValueError(f"unsupported num_slices={num_slices}")
        return _bench(fn)
    finally:
        set_sgemm_config_override(None)


# ---------------------------------------------------------------------------
# Config saving
# ---------------------------------------------------------------------------


def sort_config(config: Dict[str, Any]) -> Dict[str, Any]:
    ordered = {}
    for key in ["BLOCK_S", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"]:
        if key in config:
            ordered[key] = config[key]
    return ordered


def save_config(configs, kernel, K, rank, num_slices) -> str:
    filename = get_sgemm_config_file_name(kernel, K, rank, num_slices)
    version_dir = f"triton_{triton.__version__.replace('.', '_')}"
    config_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "..",
            "python", "sglang", "srt", "lora", "triton_ops", "sgemm_configs",
            version_dir,
        )
    )
    os.makedirs(config_dir, exist_ok=True)
    filepath = os.path.join(config_dir, filename)
    with open(filepath, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")
    return filepath


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def tune_one(
    kernel: str,
    K: int,
    rank: int,
    num_slices: int,
    max_lens: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, tuple]]:
    print(f"\n{'='*80}")
    print(f"Tuning {kernel} — K={K}, R={rank}, S={num_slices}")
    print(f"{'='*80}")

    if kernel == "sgemm_a":
        search = get_shrink_search_space(rank, num_slices)
        bench = benchmark_shrink
    else:
        search = get_expand_search_space(rank)
        bench = benchmark_expand
    print(f"Search space: {len(search)} configs")

    best_configs: Dict[int, Dict[str, Any]] = {}
    results: Dict[int, tuple] = {}
    for max_len in max_lens:
        batch_info = build_batch_info(max_len, rank, device)
        best_config, best_time = None, float("inf")
        for i, config in enumerate(search):
            t = bench(config, K, rank, num_slices, batch_info, device, dtype)
            if t is not None and t < best_time:
                best_time, best_config = t, config
            if (i + 1) % 50 == 0:
                print(
                    f"  max_len={max_len}: {i+1}/{len(search)} tested, "
                    f"best={best_time:.4f}ms"
                )
        if best_config is None:
            print(f"  max_len={max_len}: NO valid config (all launches failed!)")
            continue
        best_configs[max_len] = sort_config(best_config)
        results[max_len] = (best_time, best_configs[max_len])
        print(
            f"  max_len={max_len}: best={best_time:.4f}ms config={best_configs[max_len]}"
        )
    return best_configs, results


def get_dims(args) -> List[int]:
    """The list of K dims to tune (shrink input_dim / expand per-slice output)."""
    if args.hidden_sizes:
        return list(dict.fromkeys(args.hidden_sizes))
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", hidden_size // num_heads)
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    dims = [hidden_size, q_dim, max(q_dim, kv_dim)]
    print(
        f"Model {args.model}: hidden={hidden_size}, q_dim={q_dim}, kv_dim={kv_dim}"
    )
    return list(dict.fromkeys(dims))


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    dtype = DTYPES[args.dtype]
    ranks = args.ranks
    slices = sorted(set(args.slices))
    max_lens = list(dict.fromkeys(args.decode_batch_sizes + args.prefill_num_tokens))
    dims = get_dims(args)

    print(
        f"\nLoRA sgemm tuning: dims={dims}, ranks={ranks}, slices={slices}, "
        f"max_lens={max_lens}, dtype={args.dtype}"
    )

    all_results = []
    for K in dims:
        for rank in ranks:
            for S in slices:
                for kernel in ("sgemm_a", "sgemm_b"):
                    best_configs, results = tune_one(
                        kernel, K, rank, S, max_lens, device, dtype
                    )
                    if not best_configs:
                        continue
                    path = save_config(best_configs, kernel, K, rank, S)
                    print(f"  Saved: {path}")
                    all_results.append((kernel, K, rank, S, results))

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    print(
        f"\n{'kernel':<9}{'K':>7}{'R':>5}{'S':>3}{'max_len':>9}{'best(ms)':>11}  config"
    )
    print("-" * 100)
    for kernel, K, rank, S, results in all_results:
        for ml in max_lens:
            if ml in results:
                t, cfg = results[ml]
                print(f"{kernel:<9}{K:>7}{rank:>5}{S:>3}{ml:>9}{t:>10.4f}m  {cfg}")
    print(f"\nDone at {datetime.now().ctime()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Auto-tune dense-attention LoRA sgemm A/B kernel block sizes"
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model to derive attention dims (hidden_size, q_dim, kv_dim).",
    )
    p.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Explicit K dims to tune (shrink input_dim / expand per-slice output). "
        "Overrides --model. e.g. Qwen3.5: 2048 512; Kimi-K2.5: 7168 2048.",
    )
    p.add_argument(
        "--ranks", type=int, nargs="+", required=True, help="LoRA ranks, e.g. 16 32 64"
    )
    p.add_argument(
        "--slices",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="num_slices to tune: 1=o_proj/down, 2=gate_up, 3=qkv (default: 1 2 3)",
    )
    p.add_argument(
        "--decode-batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 48, 64],
        help="Decode max_len regimes (= batch sizes).",
    )
    p.add_argument(
        "--prefill-num-tokens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
        help="Prefill max_len regimes (= token counts).",
    )
    p.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    p.add_argument("--seed", type=int, default=None)
    # Accepted for CLI-compatibility with the MoE tuner; unused for the dense path.
    p.add_argument("--num-experts", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--top-k", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.model and not args.hidden_sizes:
        p.error("Either --model or --hidden-sizes is required")
    main(args)
