# MLA `kv_b_proj` dense-gemm experiment — status

**Branch:** `lora-mla-stepab-dense-gemm` (off `jybsuper:nvfp4-lora` @ `ac0fa6d3ee`) · **PR:** jybsuper/sglang#16
**Last updated:** 2026-06-02 ~22:10 KST · **HEAD:** `1f0a4d9076`

## Objective

Profiling flagged the four per-head Triton SGMM step kernels for the absorbed-MLA `kv_b_proj` LoRA
correction (`_step_a_q` / `_step_b_q` / `_step_a_v` / `_step_b_v` in `triton_ops/kv_b_lora_absorbed.py`)
as a non-overlapped ~45 µs attention-side hotspot on Kimi (MLA). Question: does a plain dense
`torch.bmm`/`matmul` (cuBLAS) beat the fused-but-tiny (rank ~16–32) Triton tiles? Gated by a new env
flag `SGLANG_OPT_MLA_LORA_DENSE_GEMM` (default off).

---

## 1. What I did (chronological, precise)

1. **Implemented the dense path** (`deepseek_mla_correction.py`), commit `2b2dd693c1`:
   - New env flag `SGLANG_OPT_MLA_LORA_DENSE_GEMM` (`EnvBool`, default off) in `environ.py`.
   - Dense helpers `_dense_step_a_q/_b_q/_a_v/_b_v` (torch `bmm`/`matmul`), `_slot_weights` (graph-safe
     length-1 `index_select`), `_dense_path_active` gate.
   - Wired into BOTH the serial `apply_q/v_correction` and the two-stream `kv_b_lora_q/v_prepare→apply`
     paths; Triton fallback otherwise.
2. **CPU numerical-equivalence test** (`test_dense_mla.py`) vs the kernel reference math: **ALL_MATCH**
   (max err ~1e-14). *(Used float64 — did NOT catch the bf16 dtype bug found later.)*
3. **Brought up 2-node Kimi-K2.5-NVFP4 pods** (k8s `ID=ys-mla-0602-1754`, `--tp 8` MNNVL, trtllm LoRA
   backend + virtual experts, adapter `alpha` which targets `kv_b_proj`, `SGLANG_LORA_TWO_STREAM=1`).
   - A/B = **identical commit**, base = flag OFF vs variant = flag ON (numerically-equivalent → acc
     diff must stay within the ~0.30 atomic-add noise floor).
4. **Opened PR #16**, committed this journal into the branch, filled done/to-do + perf/acc tables.
5. **Infra issues debugged & fixed along the way (NOT my LoRA code):**
   - 2 h "Pending": I had set ephemeral-storage request to 600 Gi, but k8s reports every node's
     ephemeral allocatable as ~123 Gi (bogus; real container disk is 1.8 T) → unschedulable. Reverted
     to 100 Gi → scheduled immediately.
   - `run_kimi.sh` `wait_ready` DIED-check matched the pod entrypoint string `sglang` (always ≥1) →
     would never detect a dead server. Fixed to match `launch_server` + also check the worker node.
   - `run_kimi.sh` `pull_traces` `local cell=$1 … src="${cell}"` crashed under `set -u` on bash 3.2
     (later var in one `local` can't see an earlier one). Split the declaration.
6. **First full run completed** acc+bench for both cells. summary.py: acc mean|Δ|=0.2685 (≤0.30),
   perf variant=100.3/100.7/100.6 % of base. **This looked like a tiny +0.3–0.7 % win — but it was
   INVALID.**
7. **Caught the no-op via traces:** base and variant had **identical** `_step_*` kernel counts (671
   each) → `SGLANG_OPT_MLA_LORA_DENSE_GEMM=1` had **zero effect**. Root cause: `_dense_path_active`
   rejected `permutation is not None`, but the Triton backend always sets `permutation =
   argsort(weight_indices)` (identity for a single adapter). **Fix** `9d8c2927c4`: gate on
   `num_segments==1` only (single slot ⇒ permutation is a no-op for per-row dense).
8. **Re-ran variant with dense actually engaged → it CRASHED during cuda-graph capture:**
   `_dense_step_b_q … RuntimeError: expected mat1 and mat2 to have the same dtype, BFloat16 != float`.
   `A * scaling` promoted bf16 `A` by fp32 `scaling` to float32. **Fix** `1f0a4d9076`: apply scaling
   to the matmul *result* (fp32) and cast back, keeping matmul operands bf16 (matches the kernel's
   fp32-accumulate-then-cast). Affects `_dense_step_b_q` and `_dense_step_b_v`.

## 2. Current status (as of HEAD `1f0a4d9076`)

- **Code:** dense path implemented + 2 correctness fixes (permutation gate, bf16 dtype). Pushed;
  PR #16 reflects HEAD.
- **Valid results so far:** only the **base baseline** (Triton kv_b), bench (in=out=2048):

  | bs | output tput (tok/s) | ITL (ms) |
  |---|---|---|
  | 16 | 672 | 23.82 |
  | 32 | 1210 | 26.45 |
  | 64 | 2080 | 30.77 |

- **No valid dense-vs-Triton numbers yet** — every variant run so far was either a no-op (gate bug) or
  crashed (dtype bug). Both bugs are now fixed but the fixed variant has **not** completed a run.
- **Pods:** still up (`mnnvl-kimi-ys-mla-0602-1754-0/1`), GPUs idle/clean.

## 3. Next steps

1. Rebuild + re-inject the variant bundle at `1f0a4d9076`; re-run the **variant cell only** (base
   results are valid and reused).
2. **Verify dense actually engaged** via traces: `_step_a_q/_b_q/_a_v/_b_v` counts should drop to ~0 in
   variant (replaced by bmm), unlike the no-op run.
3. **Accuracy:** `summary.py` — dense vs base logprob |Δ| must stay within the ~0.30 noise floor
   (also guards the cuda-graph padding-row question, since dense processes padded rows the kernel masks).
4. **Perf verdict:** `decode_isolate.py` — decode-isolated kv_b kernel time, dense bmm vs Triton
   `_step_*`. This is the real signal (E2E delta is small; kv_b is a few µs/layer).
5. Decide keep-vs-drop the flag based on (3)+(4); update PR. Release pods. (Secondary: MoE shared
   experts dense gemm — only if the step kernels show a win.)

> Detailed chronological log (incl. every command) lives in
> `river/task-lora-mla-stepab-dense-gemm/journal.md` (local). This file is the PR-facing summary.
