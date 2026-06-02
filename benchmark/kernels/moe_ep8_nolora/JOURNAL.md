# Journal — MoE EP8 vs TP8 (no-LoRA baseline) bench & profile

- **Date started:** 2026-06-02
- **Owner:** yushengsu
- **Task branch:** `moe-ep8-nolora-bench` (based on `lora-opti-nvfp4` @ `ac0fa6d3ee`, tracking `jybsuper/nvfp4-lora`)
- **Worktree:** `/Users/yushengsu/Downloads/river/sglang-moe-ep8-nolora-bench`
- **Model under test:** `nvidia/Kimi-K2.5-NVFP4` (2 nodes, 4+4 GPU MNNVL/GB200), 384 experts, hidden 7168, LoRA rank 16/32.

---

## Goal

Find out whether running the MoE under **EP8** (expert parallel, each rank owns 384/8 = 48 experts)
instead of the current **TP8** (every rank holds the full expert weight, GEMM rhs not sliced) avoids
the memory-bound LoRA "tax" we measured under TP, **without** hurting the no-LoRA baseline too much —
especially at **larger batch sizes**.

This first task is the **best no-LoRA baseline only**: switch MoE TP8 → EP8 and measure the speed
difference (bench + profile). If EP8 is acceptable at larger bs, EP becomes the foundation for moving
the LoRA MoE kernels onto a 1/8-sliced (per-rank 48-expert) layout.

## Motivation / context (from the P0 thread)

Under **TP8**, two of the four grouped MoE-LoRA GEMMs do **not** get their rhs sliced, so each rank
streams the full per-expert weight — pure memory traffic, overlap can't hide it because we already
assume 8 TB/s is saturated:

| # | GEMM (grouped, per-expert) | TP8 (rhs) | EP8 |
|---|---|---|---|
| ① | gate_up lora_A (shrink, H→2r) | `[384,M,7168]×[384,7168,32]` — **rhs full, K=7168 not sliced** | `[48,M,7168]×[48,7168,32]` |
| ② | gate_up lora_B (expand, 2r→2l) | `[384,M,16]×[384,16,512]` (block-diag, N=4096/8) | `[48,M,16]×[48,16,4096]` |
| ③ | down lora_A (shrink, l→r) | `[384,M,256]×[384,256,16]` (K=2048/8) | `[48,M,2048]×[48,2048,16]` |
| ④ | down lora_B (expand, r→H) | `[384,M,16]×[384,16,7168]` — **rhs full, N=7168 not sliced** | `[48,M,16]×[48,16,7168]` |

Theoretical TP8 tax @8 TB/s: gate_up A ≈ 21us, down B ≈ 10.5us (84 MB) — unavoidable, no overlap help.
EP8: every kernel's compute/traffic ÷8 → as little as ~4us. Profile (DECODE bs=64) confirms the two
"not sliced" kernels (`_moe_lora_*` circled) are visibly the slow ones; the 8-way-sliced kernels are
much faster by eye.

**Hypothesis:** since bs is tunable, at larger bs EP8's all-to-all + per-rank compute should amortize
well and EP8 should not be much worse than TP8 on the no-LoRA baseline — while killing the LoRA tax.

## Plan / actionable items

1. **[no-LoRA only] MoE TP8 → EP8** on the Kimi-K2.5-NVFP4 best baseline; measure speed delta.
   - Baseline (control): current `--tp 8` (no EP), no-LoRA, default MoE backend.
   - Variant: `--tp 8 --ep-size 8` + appropriate `--moe-a2a-backend` / `--moe-runner-backend`
     for NVFP4 cross-node EP (`ep_size ∈ {1, tp_size}`; flashinfer_cutlass/cutedsl support EP on fp4).
   - Sweep **larger bs** (not just 16/32/64 — add 128/256, bump `--cuda-graph-max-bs`) to test the
     "EP fine at large bs" hypothesis.
2. **If EP8 looks bad due to expert imbalance (balancedness):**
   - `--init-expert-location <dist>` to flatten the expert placement, or
   - drop to **4 GPU** (tp=ep=4) which should be more balanced.
3. Profile both (graph-on bs16/64[/128/256], graph-off bs16) and compare the circled kernels +
   end-to-end decode step time. Pull traces locally.

## Conventions

- k8s id: `yushengsu-<date>-<time>` (avoid resource conflicts).
- Per skill.md: regression = `sglang-base-variant-regression.md` (Qwen) / `kimi-regression` (Kimi);
  perf = `sglang-lora-base-perf-benchmark.md`. Release nodes when done.
- This task touches **launch flags / bench config** primarily (no-LoRA EP8 should work on stock code);
  any sglang code fix needed to make NVFP4 EP8 launch will be recorded below and committed.

---

## Log

### 2026-06-02 — setup
- Confirmed base branch `lora-opti-nvfp4` in sync with `jybsuper/nvfp4-lora` (`0 0`).
- Created worktree `sglang-moe-ep8-nolora-bench` + branch `moe-ep8-nolora-bench` off `ac0fa6d3ee`.
- Verified EP server-args in this build: `--ep-size` (∈{1,tp}), `--moe-a2a-backend`
  {none,deepep,flashinfer,…}, `--init-expert-location` (default `trivial`).
- Next: author the EP8 no-LoRA launch variant of `run_kimi.sh`, commit journal + script, open PR,
  then launch the 2-node Kimi pod and run.
