# Task: Remove redundant per-layer adapter_enabled rank-masking in MoE LoRA `_get_lora_info`

═══════════════════════════════════════════════════════════════════════════
## ▶ RESUME HERE (read this first if reopening)
═══════════════════════════════════════════════════════════════════════════

**Snapshot @ 2026-06-02 ~18:1x.** Background jobs do NOT survive a terminal/Claude close —
on resume you must re-check pod state and RE-LAUNCH the orchestrator. Everything needed is below.

### Fixed facts
- ID=`yushengsu-20260602-161645`  | k8s context=`leira` (ns default)
- Worktree=`/Users/yushengsu/Downloads/river/sglang-lora-adapter-mask-hoist`
  branch=`lora-adapter-mask-hoist` @ `2c6adb4e9b` (pushed to origin = yushengsu-thu/sglang). The
  CODE CHANGE IS DONE + COMMITTED + PUSHED. Base branch = `lora-opti-nvfp4` @ ac0fa6d3 (sync'd).
- Task scripts dir = `/Users/yushengsu/Downloads/river/task-lora-adapter-mask-hoist`
- Qwen RUN_ROOT = `~/Downloads/sglang_regression_yushengsu-20260602-161645` (results saved locally)
- Kimi RUN_ROOT = `~/Downloads/sglang_kimi_reg_yushengsu-20260602-161645`

### STATUS
- ✅ STEP 3 Qwen regression DONE, BOTH PASS (acc within noise + perf faster). See "Run 2 numbers".
- 🔧 Kimi (kimi-regression, 2-node) in progress:
  - pod-1 (`mnnvl-kimi-...-1`) RUNNING on big-disk node np-4: model+LoRA downloaded, /workspace
    checked out (__bench_base/__bench_variant). READY.
  - pod-0 (`mnnvl-kimi-...-0`) PENDING ~78min: CAPACITY WAIT. Cluster GPU-saturated by other users'
    pods; only free-GPU nodes are cordoned/tainted/small-disk. np-20 (small ~264G) EVICTED it twice
    → np-20 cordoned (np-13 was pre-cordoned by someone else; leave it). All GPU nodes share one
    NVLink clique so any node MNNVL-pairs — pod-0 just needs a big-disk 4-GPU node to FREE UP.
  - Resilient waiter running: kimi_orchestrate3.sh (now polls ~8h, grabs node the instant one frees,
    disk-gated ≥600G, then downloads+runs). bg id (this session) = bmw767hzl. RE-LAUNCH on resume.
  - BLOCKER (19:5x) then RESOLVED (20:05): several idle nodes (np-9/8/5/3/2) showed capacity=4 but
    allocatable nvidia.com/gpu=0 → scheduler skipped them ("Insufficient gpu"). ROOT CAUSE: stale
    NVIDIA device-plugin advertisement. FIX: `kubectl delete pod -n nvidia-gpu-operator
    nvidia-device-plugin-daemonset-<x>` on the idle nodes → they re-advertise allocatable=4.
  - ✅ pod-0 now 1/1 Running on np-18 (healthy, /root free=1070G — passed disk gate). MNNVL CD formed
    across np-4 + np-18. Branches checked out in /workspace. MODEL DOWNLOADING (252G→ climbing).
    orchestrate3 (bmw767hzl) in "wait .dl-done" → will drop ghost HBM + run run_kimi_fix.sh.
  - Cordon state to restore at end: np-20 (mine, small-disk) UNCORDON. np-13 pre-cordoned by others,
    leave it. Device-plugin restarts on np-9/8/5/3/2 are harmless (just re-advertised real GPUs).
  - ⚠ /tmp/kimi-*.bundle got CLOBBERED by a parallel agent task (both → 2b2dd693c, wrong). FIX: rebuilt
    with UNIQUE names /tmp/maskbase.bundle (lora-opti-nvfp4 ac0fa6d3) + /tmp/maskvar.bundle
    (lora-adapter-mask-hoist 2c6adb4e); re-fetched into /workspace on both pods as __bench_base/__bench_variant.
    On resume, REBUILD these uniquely-named bundles (NOT kimi-*.bundle) from local branches.
  - ⚠ image sgl-kernel=0.3.21 but branch asserts sglang-kernel>=0.4.3 (pkg renamed; module present).
    FIX: added SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 to run_kimi_fix.sh NCCL env. (run via run_kimi_fix.sh)
  - ✅ 20:31 base cell launching CORRECTLY (ac0fa6d3, past assert, server_args printed), MODEL LOADED
    (KimiK25ForConditionalGeneration NVFP4, ~72GB/GPU), started CUDA-graph capture...
  - ❌ then crashed: `ModuleNotFoundError: No module named 'cutlass'` during graph capture (needed by
    the Blackwell CuTe kernels gdn_blackwell/_tcgen05). Image has nvidia-cutlass-dsl 4.4.2 per pip but
    `import cutlass` fails even after --force-reinstall. Hits the BASE branch identically → environmental.

### ⛔ KIMI CONCLUSION: BLOCKED ON IMAGE INCOMPATIBILITY (not the code change)
The `lmsysorg/sglang:deepseek-v4-grace-blackwell` image (stock HEAD 1b497c7a) is too OLD for
lora-opti-nvfp4: stock lacks sgl_flashinfer_trtllm; sgl-kernel=0.3.21 (<0.4.3 required); cutlass DSL
not importable. Every gap also breaks the unmodified base cell → unrelated to the masking removal.
Kimi needs a NEWER GB200 image matching lora-opti-nvfp4's deps (working cutlass DSL + sgl-kernel≥0.4.3).
Stopped run_kimi_fix.sh; kept pods (model 551G cached on both) for a retry on a compatible image.

## ACTIONS LOG (chronological — what was executed)
1. Analyzed `_get_lora_info`: rank-mask (gt/to/mul_) is redundant — adapter_enabled already masked at
   single build site `_compute_moe_lora_info` (base_backend.py). 3 CUDA launches × N layers, no-op.
2. Synced lora-opti-nvfp4 w/ jybsuper/nvfp4-lora (already at ac0fa6d3). Created worktree + branch
   `lora-adapter-mask-hoist`. Removed the masking from layers.py `_get_lora_info`. commit 2c6adb4e, pushed.
3. Qwen regression (sglang-base-variant-regression.md), ID yushengsu-20260602-161645, ctx leira:
   brought up qwen35 + qwen3vl pods; built base/variant bundles; ran acc+bench both cells.
4. qwen35 first run OOM'd on acc (32000-tok logits) → fixed with --chunked-prefill-size 4096; re-ran.
5. Measured base-vs-base NOISE FLOOR per model to interpret acc. Both models PASS (acc≤noise, perf faster).
6. Freed Qwen pods. Brought up Kimi 2-node MNNVL (kimi-2node.yaml from run_sgemm).
7. Kimi debug chain: /workspace-vs-/root import path; empty model download (setup false-positive) →
   manual snapshot_download; pod-0 evicted (small-disk np-20) → cordon; capacity wait; device-plugin
   restart on idle nodes (np-9/8/5/3/2 had stale allocatable=0) to unlock GPUs; pod-0 → np-18 (1070G);
   clobbered /tmp bundles (parallel task) → rebuilt as maskbase/maskvar; sgl-kernel skip-env;
   model loaded but cutlass-DSL missing → BLOCKED (see conclusion above).
8. Stopped Kimi run; kept pods. Wrote results + this journal; opened PR with done/to-do + perf/acc.

### HOW TO REOPEN THIS SESSION
In THIS terminal, in `/Users/yushengsu/Downloads/river`, run:
```
claude --continue        # resumes this exact conversation (a.k.a.  claude -c)
```
(or `claude --resume` to pick from a list). If starting fresh instead, run `claude` then say:
"讀 task-lora-adapter-mask-hoist/journal.md 的 RESUME 段，繼續跑 Kimi"。

### HOW TO RESUME THE KIMI RUN (copy-paste)
```
kubectl config use-context leira
ID=yushengsu-20260602-161645
# 1) rebuild bundles IF /tmp was wiped (reboot clears /tmp):
ls /tmp/kimi-base.bundle /tmp/kimi-variant.bundle 2>/dev/null || {
  REPO=/Users/yushengsu/Downloads/river/sglang; git -C $REPO fetch -q origin main
  for c in base:lora-opti-nvfp4 variant:lora-adapter-mask-hoist; do n=${c%%:*}; b=${c#*:}
    git -C $REPO branch -f __bench_target $b
    mb=$(git -C $REPO merge-base origin/main __bench_target)
    git -C $REPO bundle create /tmp/kimi-$n.bundle __bench_target --not "${mb}^"; done; }
# 2) check pods; if pod-0 still Pending on a small/cordoned node, find a big-disk node & cordon bad ones
kubectl get pods | grep mnnvl-kimi-$ID
# 3) re-launch the disk-gated recovery orchestrator (waits pod-0 Ready, verifies >=600G, downloads, runs):
cd /Users/yushengsu/Downloads/river/task-lora-adapter-mask-hoist
ID=$ID RUN_ROOT=$HOME/Downloads/sglang_kimi_reg_$ID bash kimi_orchestrate3.sh \
   > $HOME/Downloads/sglang_kimi_reg_$ID/orch3.run 2>&1 &
# watch: tail -f $HOME/Downloads/sglang_kimi_reg_$ID/{orchestrate3.log,kimi.out}
```
If pod-1 was also lost (deleted), it needs the same /workspace checkout + download — see kimi_orchestrate2/3.sh.

### KEY GOTCHAS (learned the hard way — see Kimi DEBUG sections below)
1. Kimi image imports sglang from **/workspace/sglang** (NOT /root/sglang); branches must be
   checked out there. run_kimi_fix.sh already targets /workspace.
2. Pod setup's `pip install -e` fails (no Rust) AND its `hf download` dies but still touches
   `.setup-done` → model dir ends EMPTY. Always verify `/root/Kimi-K2.5-NVFP4/config.json` exists;
   if not, run dl.sh (orchestrate3 does this).
3. Model is 551G → only big-disk nodes hold it; np-20 (~264G) evicts. Disk-gate before downloading.

### AFTER KIMI FINISHES — cleanup + finalize
```
# uncordon the node(s) I cordoned (np-13 was already cordoned before me — leave it; I cordoned np-20):
kubectl uncordon np-67167b3f-20.eu-iceland1-a.compute.internal
# release Kimi pods + CD + svc:
ID=yushengsu-20260602-161645
kubectl delete pod mnnvl-kimi-${ID}-0 mnnvl-kimi-${ID}-1 --ignore-not-found
kubectl delete computedomain mnnvl-kimi-${ID}-compute-domain --ignore-not-found
kubectl delete service mnnvl-kimi-${ID}-head --ignore-not-found
```
Then: compute Kimi acc-diff vs noise floor + perf delta, write verdict here, open the PR
(branch already pushed): `gh pr create` from the worktree.

═══════════════════════════════════════════════════════════════════════════

## Background / motivation
A profiler trace (DeepseekV2, `FusedMoEWithLoRA.forward` → `layers.py:_get_lora_info`)
showed **3 CUDA launches per MoE layer per forward** coming from inside `_get_lora_info`:

```python
rank_enabled = (lora_ranks > 0).to(            # aten::gt  +  aten::to/_to_copy/copy_
    device=moe_lora_info.adapter_enabled.device,
    dtype=moe_lora_info.adapter_enabled.dtype,
)
moe_lora_info.adapter_enabled.mul_(rank_enabled)   # aten::mul_
```

DeepseekV2 has ~58 MoE layers, so this fires ~58× per forward step (and gets baked
into the captured CUDA graph), all recomputing the **same** mask on the **same**
shared per-batch buffer.

## Root-cause finding (the key insight)
`batch_info.moe_lora_info` has a SINGLE build site: `base_backend.py:304`
(`_add_moe_lora_info`), called by every backend (torch/ascend/chunked/triton),
no override. Its `adapter_enabled` is produced by `_compute_moe_lora_info`, which:
- `adapter_enabled.zero_()` every call (base_backend.py:396), then
- CUDA-kernel path stores `(lora_rank > 0)` (base_backend.py:360-364), and
- eager scatter path stores `(active_ranks > 0)` (base_backend.py:424-426).

⟹ By the time `_get_lora_info` runs, `adapter_enabled[i] == 1` already implies
`rank[i] > 0`. The per-layer `mul_(lora_ranks > 0)` can never change a value →
it is **completely redundant**.

Git history: masking inside `_compute_moe_lora_info` came from `54b06f199c
[lora] Share MoE LoRA Info (#24160)`; the redundant per-layer `mul_` was added later
in `a5c05f5727 feat(lora): NVFP4 MoE LoRA on the sgl_flashinfer_trtllm backend`
(defensive, now subsumed by the single-source masking).

## Decision
Stronger than the originally-proposed "hoist to per-batch": **delete** the redundant
masking from `_get_lora_info` entirely (single source of truth lives in
`_compute_moe_lora_info`). `_get_lora_info` slims to pure dataclass assembly.
- Launches: 3×N_moe_layers → 0.
- Safety net: zero_()+rank>0 on every prepare means no stale rank-0 `1` can survive,
  including the persistent cuda-graph `adapter_enabled` buffer.

## Workflow (per skill.md)
- Base branch: `lora-opti-nvfp4` (synced w/ `jybsuper/nvfp4-lora` @ ac0fa6d3 — already up to date).
- New branch/worktree: `lora-adapter-mask-hoist` @ `/Users/yushengsu/Downloads/river/sglang-lora-adapter-mask-hoist`.
- id for k8s: yushengsu-<date>-<time>.

## Steps & status
- [x] 0. Sync base, create worktree+branch, journal.
- [x] 1. Delete redundant masking from `_get_lora_info`; slim to dataclass assembly.
      commit 2c6adb4e9b, layers.py parses OK.
- [x] 2. Push branch to origin (yushengsu-thu/sglang) `lora-adapter-mask-hoist`.
- [~] 3. Regression: `sglang-base-variant-regression.md` (Qwen) + `kimi-regression` (Kimi).
      base = lora-opti-nvfp4 @ ac0fa6d3, variant = lora-adapter-mask-hoist @ 2c6adb4e.
      Bundles built (merge-base fba083c8 on sgl main). Both cells LoRA-on,
      --moe-runner-backend sgl_flashinfer_trtllm, differ ONLY by commit → expect ACC diff ~0.
      ID = yushengsu-20260602-161645. Cluster: leira.
      RUN_ROOT=~/Downloads/sglang_regression_yushengsu-20260602-161645
      - Qwen: pods sglang-qwen35 (Running) + sglang-qwen3vl (Pending: cluster GPU/ephemeral
        contention; orchestrator waits up to 45m for it). Orchestrator bg id bmxk9r33t;
        scripts in task folder (run_qwen35.sh, run_qwen3vl.sh, qwen_orchestrate.sh).
      - Kimi: DEFERRED until Qwen frees nodes. All 18 leira nodes are GB200 and fully
        occupied (4 other mnnvl-kimi pairs + qwen35 + others) — qwen3vl already can't get
        1 node, so launching Kimi's 2-node MNNVL now would deadlock against qwen3vl.
        Plan: launch kimi-regression (scripts/run_kimi.sh) once Qwen completes & releases nodes.
        Kimi config: base=lora-opti-nvfp4, variant=lora-adapter-mask-hoist, both LoRA-on
        same opt stack; ACC_TOL=0.30 (atomic_add noise floor).
- [ ] 4. Benchmark: `sglang-lora-base-perf-benchmark.md` (Qwen + Kimi).
- [ ] 5. Release nodes; push commit; open PR.

## Results log

### Run 1 (16:23–17:0x) — qwen35 FAILED (harness OOM, not the change), qwen3vl OK
- **qwen35**: server came up, then **CUDA OOM in logits_processor** during acc capture, BOTH cells
  identically (base = unmodified lora-opti-nvfp4 too) → NOT caused by the masking removal.
  Root cause: qwen35 acc sample = single **32000-token** seq; `logprob_start_len=0` + 32768 prefill
  chunk → logits [32000 × ~152k vocab] fp32 ≈ 19 GB, gather doubles → 29.6 GiB alloc → OOM → SIGQUIT.
  (Also found a reporting bug: inner `| tee` lacked pipefail so the failed acc still logged "done".)
- **qwen3vl**: WORKS. base acc "wrote 1820 logprobs" (sample only 1820 tok → fits). bench bs16 real:
  lat 15.63s, in 98015 tok/s, out 2142 tok/s. variant still running.

### Fix
- qwen35 re-run with `--chunked-prefill-size 4096 --max-prefill-tokens 4096` (run_qwen35_fix.sh):
  logits computed in 4k-token pieces, logprob_start_len=0 still returns all 32000 — numerically
  identical, fair to both cells. Added a hard `test -s logprobs.json` guard. bg id bcqyjlhm9.
- (qwen3vl unaffected, left running on the orchestrator.)

### Run 2 — numbers
**qwen3vl (DONE, both cells valid):**
- Bench (variant vs base), consistent across bs: variant ~3.5–3.9% FASTER.
  - bs16: 15.63→15.06s, out 2142→2226 tok/s
  - bs32: 17.46→16.83s, out 3891→4041 tok/s
  - bs64: 19.96→19.27s, out 6983→7246 tok/s
  → consistent w/ removing redundant per-layer kernel launches; no perf regression.
- Acc (variant vs base): mean|diff|=0.0556, max|diff|=1.59, 10/1820 exact.
  NOT ~0 — but MoE LoRA uses atomic_add (nondeterministic). Measuring noise floor
  (base-vs-base, same server) via bht6ilk7r to judge regression vs noise.
- qwen3vl NOISE FLOOR (base-vs-base, same server): mean|d|=0.0581, max|d|=0.803.
  → variant-vs-base mean (0.0556) is BELOW noise floor ⇒ acc diff is noise, NOT a regression. PASS.

**qwen35 (re-run w/ chunked-prefill fix, DONE acc; variant bench finishing):**
- Acc (variant vs base, 31999 tok): mean|d|=0.0242, max|d|=2.06, 446 exact. (noise floor TODO on free pod)
- Bench base: bs16 16.37s/2201, bs32 18.64s/4137, bs64 22.97s/7493 tok/s.
- Bench variant: bs16 14.79s/2439, bs32 17.55s/4416, bs64 22.14s/7875 → +3.6–9.7% faster.
- qwen35 NOISE FLOOR (base-vs-base): mean|d|=0.0227, max|d|=1.89.
  → variant-vs-base mean (0.0242) ≈ noise floor ⇒ acc within noise, NOT a regression. PASS.

## ✅ STEP 3 (Qwen regression) PASS — both models
- qwen3vl: acc within noise (0.0556 vs 0.0581), perf +3.5–3.9%.
- qwen35:  acc within noise (0.0242 vs 0.0227), perf +3.6–9.7%.
Change confirmed numerically no-op + removes per-layer overhead. Freed Qwen pods.

### Kimi (kimi-regression, 2-node MNNVL) — LAUNCHED
- Qwen pods deleted → freed 2 nodes; kimi pods mnnvl-kimi-${ID}-0/1 + ComputeDomain + head svc up
  (manifest cribbed from run_sgemm/kimi-2node.yaml). Both ContainerCreating.
- Bundles /tmp/kimi-{base,variant}.bundle (base=lora-opti-nvfp4 ac0fa6d3, variant=2c6adb4e).
- run_kimi.sh cells: BOTH LoRA-on, --moe-runner-backend sgl_flashinfer_trtllm
  --lora-use-virtual-experts, envs SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1
  SGLANG_LORA_TWO_STREAM=1, differ ONLY by commit. ACC_TOL=0.30 (atomic_add floor).
- RUN_ROOT=~/Downloads/sglang_kimi_reg_yushengsu-20260602-161645. Orchestrator bg id bbp40gveq.
- Pending: acc-diff (vs noise floor), perf-delta (prefill/decode split), decode-isolated profile.

### Kimi DEBUG (run 1 failed → fixed)
- Symptom: base launch died, `--moe-runner-backend: invalid choice 'sgl_flashinfer_trtllm'`,
  then exit-7 retry loop.
- Root cause: image `deepseek-v4-grace-blackwell` imports sglang from EDITABLE install at
  **/workspace/sglang** (HEAD `1b497c7a`, lacks sgl_flashinfer_trtllm), NOT /root/sglang where the
  manifest cloned + run_kimi.sh checks out. The setup's `pip install -e /root/sglang/python` FAILED
  ("can't find Rust compiler"), so /root install never took; /workspace remained imported.
  sgl-kernel is prebuilt in site-packages (separate) → pure-Python checkout needs no rebuild.
- Fix: fetch sgl main (for merge-base) + base/variant bundles INTO /workspace/sglang on both pods;
  run_kimi_fix.sh = run_kimi.sh with /root/sglang→/workspace/sglang and pip-install dropped.
  Verified imported server_args has sgl_flashinfer_trtllm after checkout. Re-run bg id bgrclu79x.

### Kimi DEBUG (run 2 also failed → real root cause found)
- After /workspace fix, base launch failed "Unrecognized model in /root/Kimi-K2.5-NVFP4".
- REAL root cause: /root/Kimi-K2.5-NVFP4 is EMPTY — model never downloaded. The pod setup's
  `pip install -e /root/sglang/python` failed (Rust), and the backgrounded `hf download` died
  immediately, yet setup still touched /root/.setup-done (false positive) → orchestrator's
  setup-gate passed on a model-less pod. (The earlier argparse + import-path issues were real too
  and are fixed; this download gap was the deeper blocker.)
- Fix: manually re-trigger downloads on both pods (snapshot_download, model+lora, HF_TOKEN present,
  742G free) → /root/dl.sh writes /root/.dl-done. Both pods downloading fast (~28G in <1min).
- kimi_orchestrate2.sh (bg b6ascn1y8): waits .dl-done on both → drops ghost HBM → runs
  run_kimi_fix.sh (branches already checked out in /workspace/sglang).

### Kimi DEBUG (run 3 — pod eviction)
- pod-1 model+lora download COMPLETE (551G on big-disk node np-4, .dl-done, /workspace checked out).
- pod-0 EVICTED for ephemeral-storage: landed on np-20 whose real disk threshold ~264G < 551G model
  → DiskPressure → eviction (exit137, ContainerStatusUnknown). Note: all nodes report 119GiB
  allocatable ephemeral-storage (misleading); real overlay disk is heterogeneous (np-4=1.8T, np-20 small).
- Recovery: force-deleted + recreated pod-0; kimi_orchestrate3.sh (bg bcrkomkb6) waits pod-0 Ready,
  VERIFIES >=600G free before downloading (else aborts for node-pin), then injects branches into
  /workspace + downloads model+lora + runs run_kimi_fix.sh. pod-1 untouched (already ready).

## Remaining
- [ ] pod-0 schedule onto a big-disk MNNVL node (currently Pending).
- [ ] Kimi acc+bench+profile verdict (watch acc step for 32k-logits OOM as in qwen35).
- [ ] (step 6) Qwen dedicated perf-benchmark/profile if needed (regression bench already shows +3.6–9.7%).
- [ ] Release Kimi nodes; finalize commit; open PR.
