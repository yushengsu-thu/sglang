# adapter_enabled mask-hoist — task journal (DID / NOW / NEXT)

Branch: `lora-adapter-mask-hoist` @ `2c6adb4e` (base `lora-opti-nvfp4` @ `ac0fa6d3`). PR: jybsuper/sglang#15.
Last updated: 2026-06-02 ~22:02 (local).

---

## 1. WHAT THE CHANGE IS
Removed the redundant per-layer `adapter_enabled` rank-masking in MoE LoRA
`FusedMoEWithLoRA._get_lora_info` (`python/sglang/srt/lora/layers.py`):

```python
# deleted:
rank_enabled = (lora_ranks > 0).to(device=..., dtype=...)   # aten::gt + aten::to/copy
moe_lora_info.adapter_enabled.mul_(rank_enabled)            # aten::mul_
```

It ran **3 CUDA launches per MoE layer per forward** (~58× on DeepseekV2-class; also baked into the
captured CUDA graph). It is a **no-op**: `moe_lora_info.adapter_enabled` has a single build site
(`_add_moe_lora_info` → `_compute_moe_lora_info`, `backend/base_backend.py`) that `zero_()`s it and
only sets `(rank > 0)` entries every `prepare_lora_batch` (both the Triton kernel and the eager
scatter path), incl. the persistent cuda-graph buffer. So `adapter_enabled[i]==1` already implies
`rank[i]>0`; the per-layer mask can never change a value.

## 2. WHAT WAS DONE (chronological, precise)
1. Root-caused the redundancy (single-source-of-truth in `_compute_moe_lora_info`).
2. Synced `lora-opti-nvfp4` with `jybsuper/nvfp4-lora` (already at ac0fa6d3). Created worktree+branch
   `lora-adapter-mask-hoist`. Deleted the masking from `_get_lora_info`. commit `2c6adb4e`, pushed.
3. **Qwen regression** (skill: sglang-base-variant-regression), ID `yushengsu-20260602-161645`, ctx
   `leira`. base=`lora-opti-nvfp4`@ac0fa6d3, variant=this branch@2c6adb4e; BOTH cells LoRA-on,
   `--moe-runner-backend sgl_flashinfer_trtllm --lora-use-virtual-experts`, differ ONLY by commit.
   - qwen35 first acc run OOM'd (single 32000-tok sample × ~152k vocab logits) → fixed with
     `--chunked-prefill-size 4096` (numerically identical), re-ran.
   - Measured base-vs-base **noise floor** per model (MoE LoRA uses atomic-add → run-to-run nondeterminism).
4. Freed Qwen pods. Brought up **Kimi** 2-node MNNVL (kimi-2node.yaml). Hit, and fixed, a chain of
   ENVIRONMENT issues (all also break the unmodified base → unrelated to the change):
   - import path `/workspace` vs `/root`; empty model download (setup false `.setup-done`) → manual
     `snapshot_download`; pod-0 evicted on a small-disk node (np-20) → cordon; cluster GPU saturation;
     **idle nodes advertised `allocatable gpu=0`** (stale device-plugin) → restarted device-plugin to
     unlock; clobbered `/tmp` bundles from a parallel task → rebuilt as `maskbase`/`maskvar`.
   - On the original image `deepseek-v4-grace-blackwell`: too OLD — missing `sgl_flashinfer_trtllm`,
     `sgl-kernel 0.3.21` (<0.4.3), broken `cutlass` DSL. **Switched image to `lmsysorg/sglang:dev-cu13`**
     (the image my Qwen runs + other users' Kimi runs use). dev-cu13 imports `/root/sglang`, `import
     cutlass` works, sgl-kernel OK. (Note: `cutlass` is needed by Kimi's linear/GatedDelta attention
     CuTe kernels — gdn_blackwell/_tcgen05 — NOT the trtllm MoE path.)
   - run on dev-cu13 was killed mid-capture (exit137, no OOM) by a **cross-session `pkill`** from another
     of the user's Kimi tasks (`pkill -f "kubectl exec.*launch_server"`). FIX: launch via an in-pod
     `/root/_rank.sh` so the local exec line has no `launch_server` string → immune.

## 3. RESULTS SO FAR

### Qwen — PASS on both models (acc within noise floor; perf faster)
| Model | variant-vs-base mean\|Δlogprob\| | noise floor (base-vs-base) | acc verdict |
|-------|---------------------------------|----------------------------|-------------|
| Qwen3-VL-30B-A3B-FP8 | 0.0556 (max 1.59) | 0.0581 | within noise → no regression |
| Qwen3.5-35B-A3B-FP8  | 0.0242 (max 2.06) | 0.0227 | within noise → no regression |

| Model | bs | base lat | variant lat | base out tok/s | variant out tok/s |
|-------|----|----------|-------------|----------------|-------------------|
| Qwen3-VL | 16 | 15.63s | 15.06s (−3.6%) | 2142 | 2226 (+3.9%) |
| Qwen3-VL | 32 | 17.46s | 16.83s | 3891 | 4041 |
| Qwen3-VL | 64 | 19.96s | 19.27s | 6983 | 7246 |
| Qwen3.5  | 16 | 16.37s | 14.79s (−9.7%) | 2201 | 2439 (+10.8%) |
| Qwen3.5  | 32 | 18.64s | 17.55s | 4137 | 4416 |
| Qwen3.5  | 64 | 22.97s | 22.14s | 7493 | 7875 |

→ numerically a no-op; removes real per-layer overhead (bigger win on 35B with more MoE layers).

### Kimi-K2.5-NVFP4 — IN PROGRESS, no numbers yet
On dev-cu13, 2-node tp8, both cells LoRA-on trtllm + virtual-experts (+ two-stream envs), differ only
by commit. As of 22:02: model loaded (NVFP4, ~72GB/GPU ×8), **CUDA-graph capture + cold fp4_gemm
autotune** running (the ~20-min frozen-log phase). No acc/bench captured yet.

## 4. CURRENT STATE (live)
- ID `yushengsu-20260602-161645`, ctx `leira`. Pods `mnnvl-kimi-...-0` (np-18) + `-1` (np-4), Running,
  image dev-cu13, model+lora present, branches in `/root/sglang`.
- `kimi_orchestrate_dev.sh` running (drives run_kimi.sh: base cell in autotune → acc → bench → 6
  profiles → variant cell). ETA first acc ~20min; full base+variant ~50-60min.
- Cordoned by me (restore at cleanup): `np-20` (small disk). `np-13` was pre-cordoned by others — leave.

## 5. NEXT
- [ ] Let Kimi finish; compute Kimi acc-diff vs base-vs-base noise floor + perf delta (prefill/decode
      split) + decode-isolated profile; add to PR #15.
- [ ] Final cleanup: `kubectl uncordon np-67167b3f-20...`; release Kimi pods + ComputeDomain + head svc.
- [ ] (optional) tune GB200 MoE config to drop the "Using default MoE kernel config" warning.

## 6. HOW TO RESUME (if this session is lost)
```
kubectl config use-context leira
ID=yushengsu-20260602-161645
# rebuild bundles from local branches if /tmp cleared:
REPO=/Users/yushengsu/Downloads/river/sglang; git -C $REPO fetch -q origin main
for c in maskbase:lora-opti-nvfp4 maskvar:lora-adapter-mask-hoist; do n=${c%%:*}; b=${c#*:}
  git -C $REPO branch -f __mh_target $b; mb=$(git -C $REPO merge-base origin/main __mh_target)
  git -C $REPO bundle create /tmp/$n.bundle __mh_target --not "${mb}^"; done
# re-run (pods must be dev-cu13; if recreated, model re-downloads):
cd /Users/yushengsu/Downloads/river/task-lora-adapter-mask-hoist
ID=$ID RUN_ROOT=$HOME/Downloads/sglang_kimi_reg_$ID bash kimi_orchestrate_dev.sh &
```
