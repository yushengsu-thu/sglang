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

- **Code:** dense path implemented + 2 correctness fixes (permutation gate, bf16 dtype). Pushed.
- **Dense now CONFIRMED engaged** (run @ `191c862ee`): variant graph-on trace `_step_a_q/_b_q/_a_v/_b_v`
  counts = **0** (base = 671 each) → the four Triton kv_b kernels are gone, replaced by dense bmm.
  Launch + cuda-graph capture succeeded (the dtype fix held). acc + bench done.
- **PRELIMINARY numbers — DO NOT TRUST YET** (base bench is ~1 h stale, different cluster conditions;
  swings are too large/inconsistent for a few-µs/layer change → re-measuring back-to-back):

  | bs | base out tput | variant(dense) out tput | base ITL | variant ITL | e2e Δ |
  |---|---|---|---|---|---|
  | 16 | 672 | 770 | 23.82 | 20.78 | +14.7% (suspiciously large) |
  | 32 | 1210 | 1303 | 26.45 | 24.56 | +7.7% |
  | 64 | 2080 | 1995 | 30.77 | 32.07 | −4.1% (dense slower at large bs) |

  variant server-log decode thpt ≈ 1980–1996 tok/s (bs64 phase; matches its e2e 1995). Base server-log
  decode thpt was overwritten → must re-capture for a fair A/B.
- **Coherent hypothesis (unconfirmed):** dense helps small-batch decode (the kv_b step kernels were a
  fixed serial overhead) and hurts large-batch (bmm scales with rows + processes padded rows). But the
  +15% magnitude is too big to accept without a controlled re-measure.
- **Pods:** up (`mnnvl-kimi-ys-mla-0602-1754-0/1`).

## 3. Next steps

1. **Controlled re-measurement (in progress):** re-run base AND variant back-to-back on the same pods
   (eliminate the 1 h-stale-base bias), capturing for BOTH cells (a) bench e2e `--show-report` AND
   (b) **server-log decode throughput** (`gen throughput (token/s)` lines) — not e2e alone.
2. **Accuracy:** `summary.py` — dense vs base logprob |Δ| within ~0.30 noise floor (also guards the
   cuda-graph padding-row question; dense processes padded rows the kernel masks).
3. **Perf verdict:** `decode_isolate.py` — decode-isolated kv_b kernel time, dense bmm vs Triton
   `_step_*` (now that dense is confirmed engaged, the `_step_*` rows should vanish in variant).
4. Decide keep-vs-drop the flag (likely: keep at small bs, fall back to Triton at large bs?); update PR.
   Release pods. (Secondary: MoE shared experts dense gemm — only if the step kernels show a net win.)

## 4. Outcome of the torch-dense wave + pivot (2026-06-02 ~22:30)

- **Accuracy: PASS.** Dense-engaged run vs base: mean|Δlogprob| **0.27365**, p50 0.122 — within the
  ~0.30 atomic-add noise floor. Dense kv_b correction is numerically correct.
- **Perf: NOT a win — shelving the torch-bmm dense path.** It lowers to ~12+ tiny kernel launches
  (transpose/contiguous → bmm → mul → add_, ×4 steps); at decode the launch overhead dominates and
  eats any saving. E2E numbers were noisy/inconsistent (bs16 +15% / bs64 −4%, base stale). The flag
  stays default-off; not worth a controlled E2E re-measure.

### New direction (user): single-LoRA *fused* Triton classic gemm
Hypothesis: the existing `_step_*` kernels are slow because of the **multi-LoRA machinery**
(weight_indices/lora_ranks lookups, mixed-rank `N_eff` truncation, seg_indptr, permutation gather,
extra segment grid axis) + being **4 separate kernels** with an HBM round-trip for the rank-dim
intermediate. For **single LoRA** (`max_loras_per_batch==1`) all of that degenerates to constants/no-ops,
leaving a plain tiled gemm. Design:
- **2 fused kernels** (q-correction, v-correction) replacing the 4 step kernels.
- Each fuses step_a (heavy reduction qk_nope/kv_lora_rank → small rank) + step_b (rank → kv/v_head_dim);
  keep the `(BLOCK_S, rank≤32)` intermediate in SRAM (no HBM write/read), scaling+accumulate in epilogue.
- Grid (S-tiles, heads); everything else constexpr; permutation ignored (identity for single slot).
- Wins: 4→2 launches, no intermediate HBM, zero routing branches, shape-specific tiling.
- Trade-off: fusing couples A+B so the two-stream A-step/base-bmm overlap is lost — measure fused-no-overlap
  vs split-with-overlap. Gate on `num_segments==1`, else fall back to current kernels.

### Sanity bound (user): each lora gemm ≥ full-gemm speed
lora_a `(M,K)@(K,r)` and lora_b `(M,r)@(r,N)` each have FEWER FLOPs than a full `(M,K)@(K,N)` gemm
(r≪N,K), so **each should be at least as fast as a comparable full gemm**. Any `_step_*` kernel that
is SLOWER than a full gemm of comparable dims is paying the multi-LoRA tax — that's the target to remove.
Use full-gemm time as the floor in the micro-bench.

### Iteration plan: micro-bench on the Kimi pod (single GPU), real kernels + real shapes
Stop iterating via the 2-node Kimi serve (~40-50 min/round). Run a **single-GPU `do_bench` micro-bench
ON a Kimi pod** using the REAL kv_b kernels + REAL Kimi shapes (H per TP rank, qk_nope=v_head_dim=128,
kv_lora_rank=512, rank from the alpha adapter), comparing: current 4 `_step_*` kernels vs a full gemm
of comparable dims (the floor) vs the new fused single-LoRA kernel. Seconds per round. Confirm the
kernel beats the floor in isolation, THEN one Kimi E2E + accuracy pass.

> Detailed chronological log (incl. every command) lives in
> `river/task-lora-mla-stepab-dense-gemm/journal.md` (local). This file is the PR-facing summary.

## 5. Micro-bench result (on Kimi pod, single GB200, real shapes) — hypothesis CONFIRMED

`bench_kv_b_kernels.py`, per-layer decode, H=8/qk=128/kv=512/rank=16:

| op | kernel µs (S=16/32/64) | full-bmm floor µs | verdict |
|---|---|---|---|
| step_a_q (S,H,128)->16 | 18.8 / 33.6 / 19.0 | ~11-14 | **1.7-2.4x SLOWER than full** |
| step_b_q (S,H,16)->512 | 11.6 / 11.1 / 11.7 | ~11-13 | ≈floor |
| step_a_v (S,H,512)->16 | 24.4 / 23.4 / 21.4 | ~11-13 | **1.9-2.1x SLOWER than full** |
| step_b_v (S,H,16)->128 | 11.5 / 11.8 / 11.5 | ~11-13 | ≈floor |
| **4-kernel sum** | **66 / 80 / 64 µs** | | |

- **Confirmed:** the two `_step_a_*` (lora_a; large-K reduction → tiny N=rank=16) are ~2x SLOWER than the
  full base bmm despite fewer FLOPs — occupancy/launch-bound from the tiny N=16 tile + multi-LoRA machinery.
  `_step_b_*` are ≈ the floor. (cuBLAS same-size gemm is even slower — tiny-gemm dispatch overhead — so the
  full-bmm ~11-13µs is the right floor.)
- **Target:** get step_a_q/step_a_v to the full-gemm floor (~11µs) and fuse 4 steps → 2; aim 64-80µs → ~25-40µs.
- Iteration loop is now seconds (this bench on 1 GPU), not a 40-min serve.
