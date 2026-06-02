# Journal — widen LoRA-B down-proj expand BLOCK_SIZE_N to 512 for large N

Branch `lora-expand-block-n-tune` (base `nvfp4-lora`). Kernel:
`python/sglang/srt/lora/trtllm_moe/specialized_expand.py` (`_moe_lora_expand_add_kernel`).
GPU: 1× / 2× NVIDIA GB200 (leira), image `lmsysorg/sglang:dev-cu13`.

## Goal

The rank-specialized LoRA-B **expand-add** on the **down-proj** (`fuse_sum_all_reduce=True`)
profiled at ~25 µs in a decode trace — the slow LoRA-MoE kernel. Find an e2e-anchored,
correctness-preserving config win.

## Root cause / finding

The expand streams each hit expert's `[N, R]` weight once and is memory-bound: with `K=R (<=64)`
the `tl.dot` is a single MMA step with no K-loop to amortize the per-tile prologue/epilogue over.
For **large N** a wider N tile cuts the fixed per-tile cost and coalesces the long weight rows.

Important: the production config already resolves to `BLOCK_SIZE_M=16` (via
`try_get_optimal_moe_config` when server args are set; `block_m=64` is only the args-less
`_get_stage_config` fallback). So **the real untuned lever is `BLOCK_SIZE_N`** (the launcher
forces 128 for N-divisible shapes), **not** block_m.

## The change

`_invoke_moe_lora_expand_add`: when `fuse_sum_all_reduce` (down-proj) **and** `N>=4096` and
`N%512==0`, use `BLOCK_SIZE_N=512`; otherwise keep the default 128. Down-proj is non-gated, so any
divisor of N is valid. Gated gate_up (`fuse_sum_all_reduce=False`) is untouched (stays 128,
N/2-aligned). Pure tiling change → output bit-identical.

## Results — kernel GPU time (down-proj expand-add, bs=64, rank=16, GB200, block_m=16)

Production bench (`bench_expand_add_down.py --config production`, amortized true device time):

| model (N) | block_n=128 (production) | block_n=512 | verdict |
|---|---|---|---|
| Kimi-K2.5 (N=7168) | **27.3 µs** (≈ 25.1 µs e2e trace) | **20.1 µs** | **~1.35x (e2e ~1.25x, ~25%)** → widen |
| Qwen3.5-35B (N=2048) | **6.46 µs** | 8.99 µs (256→7.06) | 128 already best → keep, do NOT widen |

→ widening helps **only** large N; for N≤2048 it regresses. Hence the `N>=4096` gate.
Correctness: `--mode correctness` 6/6 PASS (block_m {16,32,64} × bs {16,64}), max_abs_err ~3e-3.

## E2E server regression (base = block_n=128 vs variant = this branch)

- **Qwen3.5-35B-A3B-FP8** (tp4/ep4, LoRA on, sgl_flashinfer_trtllm): this branch does NOT change
  Qwen (N=2048 < 4096 → unchanged path). Server A/B confirmed bit-identical behavior: per-token
  logprob diff within the run-to-run noise floor (this config is non-deterministic: atomic-add MoE
  sum-reduce + allreduce-fusion), e2e throughput within ±1.7% (noise). No regression.
- **Qwen3-VL-30B-A3B-Instruct-FP8** (tp4/ep4): same — logprob diff within noise, throughput ±1.6%.
  (Needs `--mamba-scheduler-strategy no_buffer`: the base branch's VL mamba-radix-cache-v2 path has
  a pre-existing NoneType crash with `extra_buffer`, unrelated to this kernel.)
- **Kimi-K2.5-NVFP4** (2-node MNNVL, tp8 no-EP): the model where this change is active (N=7168 →
  block_n=512). Server A/B in progress. (NVFP4 trtllm MoE LoRA requires
  `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1`.)

## Done / To-do

- [x] Kernel change + heuristic (large-N down-proj only).
- [x] Correctness (6/6 PASS) + bit-identical output.
- [x] Production-config bench: Kimi 27.3→20.1 µs (~25%); Qwen keep-128 confirmed.
- [x] Qwen3.5 + Qwen3-VL server regression: no acc / no perf regression.
- [ ] Kimi-K2.5 2-node server A/B (acc + perf) — running.
- [ ] (follow-up) consider making this a generated/tuned config entry rather than a launcher branch.
