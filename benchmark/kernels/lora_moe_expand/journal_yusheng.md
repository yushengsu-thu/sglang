# Task: optimize `_moe_lora_expand_add_kernel` (LoRA-B down-proj expand-add)

**Goal:** drive `_moe_lora_expand_add_kernel`
(`python/sglang/srt/lora/trtllm_moe/specialized_expand.py`) device time to the minimum.
Profiling showed it at **~26µs w/o overlap @ bs64** on the down-proj GEMM, while the
counterpart shrink kernel is ~5µs.

Reference / optimization basis: **jybsuper/sglang PR #12**
(`benchmark/kernels/lora_moe_expand/` — `journal_tom.md`, `JOURNAL.md`,
`theoretical_analysis.md`, `bench_expand_add_down.py`) + UT
`test/registered/lora/test_moe_lora_expand_add.py`.

## Setup
- Repo: `/Users/yushengsu/Downloads/river/sglang`, base branch `lora-opti-nvfp4`
  (tracks `jybsuper/nvfp4-lora`, head `ac0fa6d3ee` — synced, up to date 2026-06-02).
- Worktree: `/Users/yushengsu/Downloads/river/sglang-lora-moe-expand-kernel-opt`,
  branch `lora-moe-expand-kernel-opt`, **based on `pr12`** (`a8ada7b9cd` = nvfp4-lora + 8
  testbed commits) so the UT + bench + `force_block_size_n` knob are present. My kernel
  optimization commits go on top (stacked on #12 the same way #12 is stacked on #11).
- id for k8s nodes: `yushengsu-<date>-<time>`.

## Optimization basis (read from PR #12 docs)
- The kernel is **memory-bound on weight streaming** (~65 MB LoRA-B for the Kimi shape).
  Roofline floor ≈ 8.4µs (65MB / 8TB/s). Currently ~20µs at the BEST config = ~42% HBM peak.
- Per-shape numbers (1× GB200, bs=64, r=16, amortized device time):
  - Kimi-K2.5 (N=7168, 384 experts): production cfg (block_m=16, block_n=128, warps=4) ≈ **27.1µs**
    micro-bench (≈ 25.1µs e2e); BEST so far **20.1µs** at block_m=16 **block_n=512** warps=4.
  - qwen35 (N=2048, 64 experts): production cfg ≈ **6.49µs**; block_m=16 best ≈ 6.38µs.
- Known wins already documented: block_m=16 (production already uses it); **block_n 128→512
  is ~1.25–1.35× on Kimi** (the launcher forces 128 because N%128==0; not optimal).
- Theoretical headroom (theoretical_analysis.md): ~1.5–2× toward ~10–13µs by **hiding the
  K=16 load latency** (no K-loop, `num_stages=1` → no pipelining) and **reducing atomic-add
  contention** (512 per-expert deltas atomic-add into 64 output rows).

## Bottlenecks to attack (from theoretical_analysis.md §3)
1. **K=16 = single MMA, no K-loop.** Each `[bm,16]·[16,bn]` tile does minimal math between
   its B load and its atomic-add epilogue → HBM latency poorly hidden. `num_stages=1`.
2. **Atomic-add reduction contention.** 512 deltas → 64 rows; many tiles contend per row.
3. **~3976 tiny tiles** (E_hit·cdiv(N,bn) ≈ 284·14), each M_e ≤ 8 < block_m=16 (~9× M pad).

## Plan / candidate optimizations
- [P0] **Reproduce baseline** on a GB200 node: correctness 6/6 + bench production(~27µs Kimi,
  ~6.5µs qwen) + block_n=512(~20µs Kimi). Confirms the testbed before changing the kernel.
- [C1] **Inner N-loop + `num_stages>1` pipelining.** Restructure: one program per (m-block =
  expert), loop over n-tiles inside, so Triton software-pipelines the dominant B weight loads
  (prefetch next n-tile while computing current). Reuses the tiny A tile across n-tiles.
  Directly targets bottleneck #1. Expect best bandwidth-efficiency gain.
- [C2] **block_n=512** as the down-proj default (the cheap known ~25% lever); combine with C1.
- [C3] Tune `num_warps` / `num_stages` for the chosen structure.
- [C4] (stretch) reduce atomic contention — explore accumulation strategy if #1/#2 dominate.
- Each candidate: correctness UT must pass, then bench Kimi + qwen. Keep only net wins.
- After kernel win: full regression (`sglang-base-variant-regression.md` Qwen +
  `kimi-regression`) + perf benchmark (`sglang-lora-base-perf-benchmark.md`, Qwen & Kimi).

## Action log (every step)
- 2026-06-02: Read all PR #12 docs (`journal_tom.md`, `JOURNAL.md`,
  `theoretical_analysis.md`), the bench (`bench_expand_add_down.py`), the UT
  (`test_moe_lora_expand_add.py`), the kernel (`specialized_expand.py`), and the
  production call path (`virtual_experts.py` `_get_stage_config` /
  `_invoke_moe_lora_expand_add` @ line 805). Established the optimization basis above.
- 2026-06-02: Synced `lora-opti-nvfp4` ← `jybsuper/nvfp4-lora` — already up to date @ `ac0fa6d3ee`.
- 2026-06-02: Created task dir `/Users/yushengsu/Downloads/river/lora-moe-expand-kernel-opt`
  + this journal.
- 2026-06-02: `git worktree add -b lora-moe-expand-kernel-opt …-kernel-opt pr12` — worktree at
  `/Users/yushengsu/Downloads/river/sglang-lora-moe-expand-kernel-opt`, based on `pr12`
  (`a8ada7b9cd`) so UT + bench + `force_block_size_n` knob are present.
- 2026-06-02: Committed this journal into the repo and opened PR (journal = single source of truth).

## Current results

### Done
- Read all reference docs; established optimization basis + bottleneck list.
- Synced base branch; created worktree + branch + task journal.
- Brought in PR #12 testbed (UT + bench + `force_block_size_n` knob).

### To-do
- [ ] Launch GB200 node; reproduce baseline (correctness 6/6 + bench Kimi/qwen).
- [ ] C1: inner N-loop + `num_stages>1` pipelining of the B weight load.
- [ ] C2: block_n=512 down-proj default (combine with C1).
- [ ] C3: tune num_warps / num_stages.
- [ ] C4 (stretch): reduce atomic-add contention.
- [ ] Full regression (Qwen + Kimi) + perf benchmark (Qwen + Kimi); push commits/PR.

### Perf / acc — current (from PR #12 docs; NOT yet reproduced by me on a node)
All 1× GB200, bs=64, r=16, amortized true device time.

| shape | config | µs | source |
|---|---|---|---|
| Kimi-K2.5 (N=7168, 384e) | production (block_m=16, block_n=128, warps=4) | **27.1** (≈25.1 e2e) | journal_tom |
| Kimi-K2.5 | BEST so far (block_m=16, **block_n=512**, warps=4) | **20.1** | journal_tom |
| qwen35 (N=2048, 64e) | production (block_m=16, block_n=128, warps=4) | **6.49** | journal_tom |
| roofline floor (Kimi) | 65 MB / 8 TB/s | ~8.4 | theoretical_analysis |

**Acc:** UT `test_moe_lora_expand_add.py` correctness **6/6 PASS** (fp32 ref vs per-expert
bf16 atomic-add, tol 5e-2 abs; max_abs_err 2.4e-3–3.2e-3). My kernel changes must keep this.

> These are the documented baseline numbers I'm starting from. My own node runs (and any
> deltas from C1–C4) will be appended below as they complete.

## Runs
(to be filled as nodes run)
