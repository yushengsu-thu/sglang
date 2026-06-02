# Task: MLA kv_b_proj LoRA — try standard (dense) gemm for the step A/B kernels

## Goal (from user profiling)
LoRA gemms in the attention sublayer are slow. Try replacing the LoRA-specialized
Triton SGMM kernels with **standard gemm** (torch `A@B` / bmm, or flashinfer API) for the
dense (single-adapter decode) case. Some may speed up, especially the extremely slow ones.

Profiling targets:
- i. Attn sublayer LoRA A/B gemms (totally non-overlapped ~45us)
  - [P0] `_step_a_v_kernel`, `_step_b_v_kernel`  (v-side correction)
  - [P0] `_step_a_q_kernel`, `_step_b_q_kernel`  (q-side correction)
  - [P1] other kernels
- ii. MoE shared experts (~20us, highly overlapped — secondary)
- NOTE: if standard gemm works → may not need to tune the lora gemm.

## Context (verified in code 2026-06-02)
- These are the **absorbed-MLA `kv_b_proj`** correction kernels (Kimi / MLA path only;
  Qwen GQA does not use them).
- File: `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` (4 kernels).
- Caller: `python/sglang/srt/lora/deepseek_mla_correction.py`
  - `apply_q_correction`: q_nope_out += q_nope @ B_kc @ A * scaling  (step A_q → step B_q)
  - `apply_v_correction`: attn_bmm += attn_output @ A.T @ B_vc.T * scaling (step A_v → step B_v)
- Per-head SGMM with per-segment adapter routing (weight_indices / seg_indptr / permutation),
  rank ~16-32, qk_nope=v_head_dim=128, kv_lora_rank=512.

## Math per step (single slot)
- A = A_buf[slot]   (rank, kv_lora_rank)
- B = B_buf[slot]   (H*FULL_K, rank),  FULL_K = qk_nope+v_head_dim;  B_kc=B[:, :, K-half], B_vc=V-half
- step A_q: q_lora_a = einsum('shi,hir->shr', q_nope, B_kc)
- step B_q: q_nope_out += einsum('shr,rk->shk', q_lora_a, A) * scaling
- step A_v: attn_lora_a = einsum('shk,rk->shr', attn_output, A)
- step B_v: attn_bmm += einsum('shr,hjr->shj', attn_lora_a, B_vc) * scaling

## Plan
1. Add `SGLANG_OPT_MLA_LORA_DENSE_GEMM` (EnvBool, default False) in environ.py.
2. Implement a dense (torch bmm/matmul) path in `deepseek_mla_correction.py`, gated by the flag.
   - Fast path: single active adapter (num_segments == 1) → graph-safe gather of one slot,
     batched per-head matmuls. Fall back to the Triton kernels otherwise (multi-adapter,
     prefill multi-segment under cuda graph, rank mismatch).
3. Verify accuracy + measure decode latency vs the Triton kernels (Kimi).
   Acceptable to conclude "no speedup" and report.
4. (secondary) MoE shared experts dense gemm — only if step kernels show a win.

## Workflow (skill.md)
- Branch `lora-mla-stepab-dense-gemm` off `lora-opti-nvfp4` (synced w/ jybsuper/nvfp4-lora @ ac0fa6d3ee).
- Worktree: `/Users/yushengsu/Downloads/river/sglang-lora-mla-stepab-dense-gemm`.
- Per step: upload to nodes, run regression (kimi-regression, since MLA) then perf benchmark, release nodes, commit.
- k8s id: yushengsu-<date>-<time>.

---

## Log

### 2026-06-02 — setup
- Verified lora-opti-nvfp4 fully synced with jybsuper/nvfp4-lora (0 ahead / 0 behind @ ac0fa6d3ee).
- Created worktree + branch `lora-mla-stepab-dense-gemm`.
- Read the 4 kernels + caller + LoRABatchInfo + triton_backend._sgemm_info (merges single-adapter
  decode repeats into 1 segment → num_segments==1 detectable host-side).

### 2026-06-02 — implementation (commit 2b2dd693c1)
- Added `SGLANG_OPT_MLA_LORA_DENSE_GEMM` (EnvBool False) in environ.py.
- `deepseek_mla_correction.py`: added dense step helpers (`_dense_step_a_q/b_q/a_v/b_v`,
  `_slot_weights`, `_dense_path_active`) using torch bmm/matmul; wired into BOTH the serial
  `apply_q/v_correction` and the two-stream `kv_b_lora_q/v_prepare→apply` paths. Fall back to
  Triton kernels when flag off / permutation present / num_segments != 1.
- Dense math validated against the kernel docstrings (CPU einsum-vs-bmm test in test_dense_mla.py;
  no torch locally → will run it on the node before regression).
- byte-compile OK.

### 2026-06-02 — Kimi run setup
- User: don't need the optimal opt-stack, just need attn overlap (since I only changed attn).
- User confirmed the alpha adapter targets kv_b_proj → dense path will fire.
- Cells (identical commit 2b2dd693c1; env-gated): both LoRA-on, trtllm backend + virtual-experts +
  SGLANG_LORA_TWO_STREAM=1 (attn kv_b correction overlaps). base=flag OFF, variant=flag ON.
  Numerically-equivalent → acc diff must be within noise floor (≤0.30).
- ID=ys-mla-0602-1754. Pods mnnvl-kimi-ys-mla-0602-1754-0/1 applied on leira.
- Bundles built (117KB) → /tmp/kimi-{base,variant}.bundle. run_kimi.sh cell block edited.
- Next: wait pods ready + setup-done → inject bundles → run_kimi.sh (acc+bench+profile).

### 2026-06-02 17:58 — BLOCKED on cluster capacity
- Both pods Pending: leira out of free GPUs (FailedScheduling: Insufficient nvidia.com/gpu on 14-15/18 nodes).
- GPU occupants: cz-kimi (33h), nv (8h), yyh-gb200 (40h), tom (2gpu), + other concurrent yushengsu agents'
  runs (mnnvl-kimi-...-161645 [pod-0 Failed], qwen35/qwen3vl-...-165445). Did NOT touch others' pods.
- User: wait on a queue — poll periodically, report when scheduled, proceed automatically once it lands.
- Strategy: background `kubectl wait` (25m) re-invokes on exit; if still Pending, report + re-arm. Pods stay queued.

### 2026-06-02 18:18 — eviction → fixed ephemeral-storage request
- pod-0 briefly scheduled onto node np-67167b3f-20, started the ~565GB Kimi model download, then
  **Evicted (node low on ephemeral-storage)**: that node has only ~110GB allocatable ephemeral, but
  the yaml requested only 100Gi so the scheduler mis-placed it there. (Same node killed the other
  agent's 161645 kimi run.) Healthy cz-kimi/nv pods use the same 100Gi request but happened to land
  on big-disk nodes.
- Fix (per SKILL "right-size requests"): local copy /tmp/kimi-2node-ys.yaml with ephemeral-storage
  REQUEST 100Gi→600Gi (limit 1200Gi untouched); recreated both pods. Now the scheduler only considers
  nodes with ≥600GB free ephemeral → skips disk-short nodes, no eviction. Did NOT edit the shared yaml.
- Pods Pending again (waiting for a node with 4 free GPUs + ≥600GB disk). Re-armed readiness watcher.

---

## ▶ RESUME STATE (2026-06-02 18:50) — read this to continue

**Where we are:** implementation DONE + committed; blocked purely on leira GPU capacity. Pods queued
(Pending). The k8s pods live on the cluster and survive closing this terminal; only the local
background `kubectl wait` watcher dies. Resuming = reconnect, check pods, continue.

**Key identifiers**
- k8s `ID` = `ys-mla-0602-1754`  → pods `mnnvl-kimi-ys-mla-0602-1754-0` / `-1`, ctx `leira`
- `RUN_ROOT` = `/Users/yushengsu/Downloads/sglang_kimi_reg_ys-mla-0602-1754_20260602_175433`
- branch `lora-mla-stepab-dense-gemm` @ `2b2dd693c1`, worktree
  `/Users/yushengsu/Downloads/river/sglang-lora-mla-stepab-dense-gemm` (off `lora-opti-nvfp4`)
- task dir holds copies: `run_kimi.sh` (cell config edited), `kimi-2node-ys.yaml` (ephemeral 600Gi),
  `kimi-{base,variant}.bundle`, `test_dense_mla.py`, `ID.txt`, `RUN_ROOT.txt`.

**The A/B**: both cells = same commit; base `SGLANG_OPT_MLA_LORA_DENSE_GEMM` OFF, variant ON; both
LoRA-on + `SGLANG_LORA_TWO_STREAM=1` (attn kv_b correction overlaps). Numerically-equivalent → acc
diff must be ≤ ~0.30 noise floor. Bench bs 16/32/64, in=out=2048.

**Resume checklist (run from `/Users/yushengsu/Downloads/river`):**
1. `ID=ys-mla-0602-1754; kubectl --context leira get pods mnnvl-kimi-$ID-0 mnnvl-kimi-$ID-1`
   - if pods are **gone** (deleted/expired): recreate →
     `sed "s/\${ID}/$ID/g" task-lora-mla-stepab-dense-gemm/kimi-2node-ys.yaml | kubectl --context leira apply -f -`
   - if still **Pending**: keep waiting (re-arm `kubectl wait --for=condition=Ready ... --timeout=25m`).
2. When both **Ready** → wait for setup: each pod `bash -lc 'until [ -f /root/.setup-done ]; do sleep 10; done'`.
3. Inject bundles to BOTH pods (copies in task dir or rebuild from branch):
   `kubectl cp task-.../kimi-base.bundle  $POD:/root/base.bundle` ; same for variant; then in-pod
   `git fetch /root/base.bundle __bench_target:refs/heads/__bench_base` and
   `git fetch /root/variant.bundle __bench_target:refs/heads/__bench_variant`.
4. (sanity) run `test_dense_mla.py` in a pod (`python3`) — expect `ALL_MATCH`.
5. `cp task-.../run_kimi.sh /tmp/run_kimi.sh; ID=$ID RUN_ROOT=$RUN_ROOT bash /tmp/run_kimi.sh > $RUN_ROOT/kimi.out 2>&1 &`
   (uses kimi-regression scripts; `SKILL=/Users/yushengsu/Downloads/river/kimi-regression`).
6. After run: `ACC_TOL=0.30 python3 $SKILL/scripts/summary.py $RUN_ROOT` +
   `python3 $SKILL/scripts/decode_isolate.py --input $RUN_ROOT/kimi/variant/traces/graph_on --base $RUN_ROOT/kimi/base/traces/graph_on`.
7. Write findings to journal + summary.md → **release pods** (delete pods/service/computedomain for $ID)
   → push commit + open PR on `yushengsu-thu` (origin), base `jybsuper/nvfp4-lora`.

**NOTE:** Don't touch other agents' pods (cz-kimi, nv, yyh-gb200, tom, yushengsu-…-161645/165445).

### 2026-06-02 20:30 — ROOT CAUSE of the 2h wait: the 600Gi request was unsatisfiable
- k8s reports `ephemeral-storage` *allocatable* ≈ **123Gi on EVERY node** (bogus/uniform — the real
  container `/` overlay is **1.8T** physical; cz-kimi holds the 551G model there fine). So the 600Gi
  request I added could NEVER schedule on any node → that's why pods sat Pending for 2h, not capacity.
- The earlier eviction on node `-20` was a *real-disk* shortage specific to that node (~260GB real),
  not the 123Gi accounting. `-20` and `-13` are now cordoned (unschedulable).
- FIX: reverted ephemeral request 600Gi→100Gi (matches working cz/nv pods). Recreated → **scheduled
  immediately**: pod-0 on `np-67167b3f-8`, pod-1 on `np-67167b3f-5` (both free + schedulable).
- Risk: if `-8`/`-5` have small real disk like `-20`, the 551G model download could evict. Monitoring
  `df` on the host during setup; if evicted → cordon that node + recreate.
- Lesson: ignore k8s ephemeral-storage accounting on these nodes; keep request at 100Gi, rely on the
  real 1.8T overlay; only avoid nodes with small *real* disk (cordon them).

### 2026-06-02 20:45 — RUN STARTED
- Pods Ready on node-8 / node-5 (both 1.8T, 684G/798G free). setup-done in 4min (model cached).
- Bundles injected (both __bench_base/__bench_variant = 2b2dd693c). 
- CPU equivalence sanity check (test_dense_mla.py) in pod: **ALL_MATCH** (maxerr ~1e-14) → dense bmm
  numerically equals the kernel reference math.
- Launched run_kimi.sh (pid 11193) → $RUN_ROOT/kimi.out. base cell first pays the ~20min cold
  fp4_gemm autotune. Waiter armed to notify on completion. ~1.5-2h total.

### 2026-06-02 21:xx — monitoring base launch; apparent stall
- base server loaded model + passed autotune, reached CUDA-graph capture (`Capturing batches bs=64`).
- Observed stall: capture frozen at 0/12 for ~5min; GPU util 0% on all 8; **worker pod (node-5) has
  only 1/4 GPUs holding the model (others ~68MiB)** — looks like a cross-rank capture barrier stall
  (SKILL robustness #1: rank death/orphan). progress.log still empty; NO acc/bench artifacts yet.
- run_kimi.sh `wait_ready` (40min budget) will TIMEOUT then auto-retry the launch once (kill_all +
  relaunch). Letting the built-in retry handle it; continuing to monitor.
- **Results so far: NONE** (base server not yet READY). perf/acc still pending.

### 2026-06-02 21:xx — user asks: results + journal→PR + done/to-do/perf/acc in PR
- Action: pushing branch to origin (yushengsu-thu) and opening the PR now (base jybsuper:nvfp4-lora),
  with done/to-do + perf/acc=pending; committing this journal into the branch so it's in the PR diff.
