# adapter_enabled mask-hoist — task journal (DID / NOW / NEXT)

Branch: `lora-adapter-mask-hoist` @ `2c6adb4e` (base `lora-opti-nvfp4` @ `ac0fa6d3`). PR: jybsuper/sglang#15.
Last updated: 2026-06-03 00:10 (local).

> 2026-06-03 00:10 — RIGOROUS RE-TEST DONE (Qwen3-VL-30B + Kimi, both sgl_flashinfer_trtllm; 3 runs/cell;
> base 3-run spread = noise floor ≤~0.5%). Verdict: Qwen3-VL **+3–4% decode** (real, >noise, consistent,
> decode≈e2e); Kimi **decode flat** (−0.9/+2.4/−0.9 mixed); extend flat on both. See PR #15 Performance
> table for full extend/decode/e2e numbers. Supersedes the earlier single-run e2e Qwen numbers.

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

All 3 models: variant vs base, both cells LoRA-on trtllm-LoRA + virtual-experts, differ only by this commit.

Test matrix (per user, 2026-06-02): **only Qwen3-VL-30B-A3B-FP8 + Kimi-K2.5-NVFP4**, both via
`sgl_flashinfer_trtllm` MoE LoRA. (Qwen3.5-35B dropped from the matrix.) Kimi config verified:
moe_runner_backend=sgl_flashinfer_trtllm, lora_use_virtual_experts=True, "Virtual expert computation
enabled", LoRA loaded on down_proj/gate_up_proj (MoE experts) + attention.

**Accuracy — bs=1** (single-sequence per-token logprob capture). MoE LoRA uses atomic-add → run-to-run
nondeterminism, so judged vs a base-vs-base noise floor:
| Model | tokens (bs=1) | mean\|Δlogprob\| | noise floor | verdict |
|-------|---------------|------------------|-------------|---------|
| Qwen3-VL-30B-A3B-FP8 | 1820 | 0.0556 | 0.0581 (measured) | within noise |
| Kimi-K2.5-NVFP4      | 1808 | 0.2873 | ~0.26–0.30 (documented) | within noise |

**Performance — output throughput (tok/s), one table** (in/out 2048/2048). Qwen3-VL rows = bench e2e
out-tput; Kimi rows = server-log decode throughput (median gen tok/s):
| Model | bs | base tok/s | variant tok/s | Δ% | metric |
|-------|----|-----------|---------------|-----|--------|
| Qwen3-VL-30B-A3B-FP8 | 16 | 2142 | 2226 | +3.9% | e2e out |
| Qwen3-VL-30B-A3B-FP8 | 32 | 3891 | 4041 | +3.9% | e2e out |
| Qwen3-VL-30B-A3B-FP8 | 64 | 6983 | 7246 | +3.8% | e2e out |
| Kimi-K2.5-NVFP4      | 16 | 675.5 | 660.9 | −2.2% | decode |
| Kimi-K2.5-NVFP4      | 32 | 1213.6 | 1229.0 | +1.3% | decode |
| Kimi-K2.5-NVFP4      | 64 | 2098.5 | 2085.1 | −0.6% | decode |

⚠️ METRIC CAVEAT — the Qwen vs Kimi numbers are NOT directly comparable, and the Qwen perf is not rigorous:
- Qwen3-VL = **bench e2e** (prefill+decode+sched), **single run/cell, NO perf noise floor**.
- Kimi    = **decode-isolated** server-log tput, median of ~100 decode steps (stable).
So "Qwen +3.9% vs Kimi ~0%" is mostly an artifact of (a) different metrics and (b) Qwen single-run variance,
NOT a real decode speedup gap. The change removes per-layer overhead that the original profile noted was
"5us but overlapped" + 3 sub-µs graph kernels → expected decode impact ≈ 0; Kimi's flat decode is the
reliable signal. TODO to settle it: re-measure Qwen3-VL with the SAME decode-tput metric + repeats.
(Qwen3.5-35B e2e showed +5–11% single-run — even less trustworthy, and out of the agreed matrix; excluded.)

Kimi run notes (2026-06-02 22:44): run1 crashed on a run_kimi.sh bash bug (`local cell=$1 … ${cell}`
same-line expand under set -u) during profiling → fixed + ran acc+bench only. Also fixed a cross-session
pkill collision (in-pod /root/_rank.sh launcher). Kimi e2e ref: base bs16 49.98s/676, variant 51.21s/659.

## 4. CURRENT STATE (live)
- ID `yushengsu-20260602-161645`, ctx `leira`. Pods `mnnvl-kimi-...-0` (np-18) + `-1` (np-4), Running,
  image dev-cu13, model+lora present, branches in `/root/sglang`.
- `kimi_orchestrate_dev.sh` running (drives run_kimi.sh: base cell in autotune → acc → bench → 6
  profiles → variant cell). ETA first acc ~20min; full base+variant ~50-60min.
- Cordoned by me (restore at cleanup): `np-20` (small disk). `np-13` was pre-cordoned by others — leave.

## 5. NEXT
- [x] Kimi acc + decode-throughput done (2026-06-02 22:44) — no-op, no regression (above).
- [ ] (optional, rigor) measure Kimi base-vs-base acc noise floor to confirm 0.2873 is noise (relaunch
      base, 2 acc passes); documented floor 0.26–0.30 + ±2% decode already support no-op.
- [ ] (optional) decode-isolated profiler traces (re-enable prof/pull_traces in run_kimi.sh, now bug-fixed).
- [ ] Final cleanup: `kubectl uncordon np-67167b3f-20...`; release Kimi pods + ComputeDomain + head svc.

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
