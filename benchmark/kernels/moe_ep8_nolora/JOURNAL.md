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

- **Perf measurement rule (REQUIRED):** never read only the `bench_one_batch_server` e2e result.
  Also read the **decode throughput (token/s) printed in the server log** (`/tmp/server.log`:
  `Decode batch ... gen throughput (token/s): ...`) and report that per bs/variant. The e2e number
  includes prefill + scheduling; the server-log decode thpt is the steady-state decode rate that the
  TP8-vs-EP8 comparison actually hinges on.
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
- Authored `run_kimi_ep_vs_tp.sh` (no-LoRA, VARIANT=tp8|ep8, larger-bs sweep, EP backend env knobs).
- Committed JOURNAL + harness (`b62becfdb6`), pushed to `origin` (yushengsu-thu fork),
  opened **PR jybsuper/sglang#18** against `nvfp4-lora`.

---

## STATUS SNAPSHOT — 2026-06-02 (precise: done / now / next)

### What has been done so far
1. **Read `skill.md`** and the Kimi section of `sglang-lora-base-perf-benchmark.md` §3 — confirmed the
   "best no-LoRA" TP8 config is the `nvidia/Kimi-K2.5-NVFP4` 2-node (4+4 GPU MNNVL) launch at
   `--tp 8` with **no EP**, default MoE backend. This is the control we want to beat.
2. **Synced base branch:** `git fetch jybsuper` → `lora-opti-nvfp4` is `0 0` vs `jybsuper/nvfp4-lora`
   (already in sync, HEAD `ac0fa6d3ee`).
3. **Created isolated worktree + branch:** `sglang-moe-ep8-nolora-bench` / branch
   `moe-ep8-nolora-bench` off `lora-opti-nvfp4` (so it doesn't collide with other agents on the
   shared repo).
4. **Verified EP server-args exist** in this build (`server_args.py`): `--ep-size` (must be 1 or
   tp_size → EP8 is the only EP option at tp8), `--moe-a2a-backend` {none,deepep,flashinfer,…},
   `--moe-runner-backend` (flashinfer_cutedsl/cutlass support fp4 EP), `--init-expert-location`.
5. **Wrote the experiment harness** `benchmark/kernels/moe_ep8_nolora/run_kimi_ep_vs_tp.sh`:
   no-LoRA baseline only, `VARIANT=tp8` (control) vs `VARIANT=ep8`, larger-bs sweep (16/32/64/128/256
   with `--cuda-graph-max-bs 256`), EP backend choices as env vars so they can be tuned on-pod
   without re-pushing, and an `--init-expert-location` hook for the balancedness fallback.
6. **Wrote this JOURNAL**, committed both (`b62becfdb6`), pushed to `origin`, **opened PR #18**.

### Current status
- **No GPU nodes launched yet.** Nothing has run on hardware — there are **no bench or profile
  numbers yet**. Everything so far is repo/branch/harness setup + the PR.
- The EP8 backend combo (`MOE_RUNNER=flashinfer_cutedsl`, `MOE_A2A=deepep`, `DEEPEP_MODE=low_latency`)
  is a **first guess** for cross-node NVFP4 EP and is expected to need on-pod iteration.
- k8s `ID` not yet chosen (will be `yushengsu-<date>-<time>` at launch).

### What's next (in order)
1. Pick `ID=yushengsu-<date>-<time>`; apply the 2-node Kimi pod spec (`kimi-2node.yaml` from the perf
   skill §3.1), wait for setup, build+inject this branch into both pods, ghost-check + clean HBM.
2. **Run `VARIANT=tp8`** (control) — bench + profile the no-LoRA baseline at bs 16…256. Pull traces.
3. **Run `VARIANT=ep8`** — same workload. Debug the EP launch (a2a / runner backend) until it serves;
   record the working flag combo here.
4. **Compare:** EP8 vs TP8 decode step time + the two "not sliced" `_moe_lora_*` kernels, per bs.
   Confirm/deny the "EP8 fine at large bs" hypothesis.
5. **If EP8 looks bad → balancedness:** retry with `--init-expert-location <dist>`, or drop to 4 GPU
   (tp=ep=4). Record deltas.
6. Write findings + numbers into this JOURNAL and the PR description, **release the nodes**, push the
   results commit.

---

### 2026-06-02 — launch
- `ID=yushengsu-20260602-220516`. Applied `kimi-2node.yaml` (ctx `leira`). Both pods
  `mnnvl-kimi-${ID}-0/1` scheduled + Running within ~8s (nodes np-67167b3f-2 / -3, eu-iceland1-a).
- Built branch bundle: `MAIN_BASE=fba083c80f` (merge-base sgl/main ↔ branch), 13 commits,
  `/tmp/sglang-branch-moe-ep.bundle` (119K).
- In-pod setup: numactl + hf accel installed, sglang cloned + `pip install -e` done, now downloading
  `nvidia/Kimi-K2.5-NVFP4` (140 files, fp4) + `kimi_k25_lora_alpha`. Waiting for `/root/.setup-done`.
- Next on setup-done: inject bundle into both pods, ghost-check + drop HBM page cache, run VARIANT=tp8.
- 22:10 progress check: lora adapter (4 files) downloaded on both pods; base NVFP4 (140 files) at
  ~96/140 (head) / ~74/140 (worker). pip install -e already done earlier in setup. Still waiting.
- ~22:18 SETUP_DONE on both pods. Injected bundle: first fetch failed (bundle ref is
  `moe-ep8-nolora-bench`, not `__bench_target`); re-fetched `moe-ep8-nolora-bench:refs/heads/__bench_target`
  → both pods at `a20eb4601`. Ghost-check: all 8 GPUs clean (~<200 MiB / 189 GB GB200 HBM), no drain needed.
- Starting **VARIANT=tp8 (control)** via `run_kimi_ep_vs_tp.sh` (driven locally): checkout __bench_target +
  pip install -e, launch no-LoRA server, bench bs 16/32/64/128/256 (in/out 2048), profile graph-on
  bs 16/64/128/256 + graph-off bs16. Running in background.
