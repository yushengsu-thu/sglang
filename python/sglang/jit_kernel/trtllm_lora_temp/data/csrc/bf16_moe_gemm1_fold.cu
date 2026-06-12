// opt7 — bf16 in-MoE fold (route b): CUTLASS grouped GEMM replacing permute+GEMM1+activation.
//
// THIS FILE IS THE P0 SKELETON (see runs/.../opt7_design/OPT7_DESIGN.md):
//   * sgl_bf16_fold_probe()        — proves the CUTLASS 4.x include path is wired into this
//                                    JIT module (P1 uses SM100 collective builders).
//   * sgl_bf16_moe_gemm1_fold_ref  — naive reference kernel pinning the EXACT fold semantics
//                                    (gather + interleaved gate/up GEMM + half-contiguous
//                                    LoRA delta + SwiGLU). Correctness baseline for P1–P3;
//                                    never used in serving.
//
// Semantics (copied from moe::dev::activation, trtllm_fused_moe_dev_kernel.cu ~L93..L116):
//   per valid permuted row r (expert e >= 0), token t = row2token[r], expanded x = row2exp[r]:
//     acc[c]   = sum_k hidden[t,k] * W[e,c,k]          c in [0, 2I)  (interleaved g0,u0,g1,u1..)
//     x1       = acc[2h]   + delta[x*2I + I + h]       // delta 2nd (contiguous) half -> x1
//     x2       = acc[2h+1] + delta[x*2I     + h]       // delta 1st (contiguous) half -> x2
//     out[r,h] = silu(x2) * x1
//   NOTE the asymmetry: GEMM columns are pair-INTERLEAVED, the LoRA delta is HALF-CONTIGUOUS.
//
// P1 plan (not compiled yet): CUTLASS Sm100 CollectiveBuilder grouped GEMM, bf16 TN,
//   per-expert problem sizes from trtllm routing (num_tokens_per_expert / cta_idx_xy maps);
//   A = pre-permuted buffer first (isolate GEMM perf vs the 57us bmm_Bfloat16 parity bar).
// P2: custom EVT epilogue = aux delta load (row idx via permuted_idx_to_expanded_idx, two
//   half-contiguous loads) + 2:1 adjacent-column fold + silu*mul, half-width store.
// P3: gather prologue (A rows via permuted_idx_to_token_idx) — removes standalone permute.

#include "tvm_ffi_utils.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/version.h>  // CUTLASS include path probe (flashinfer/data/cutlass/include)

using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace sgl_bf16_fold {

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// P0 reference fold kernel: one thread per (permuted row, hidden idx) via grid-stride.
// Deliberately naive (2K MACs/thread, fp32 accum) — correctness baseline only.
__global__ void foldRefKernel(
    __nv_bfloat16 const* __restrict__ hidden,   // [num_tokens, K]
    __nv_bfloat16 const* __restrict__ w,        // [E, 2I, K] row-major, interleaved g/u cols
    __nv_bfloat16 const* __restrict__ delta,    // [num_expanded, 2I] half-contiguous, or null
    int const* __restrict__ row2token,          // [R]
    int const* __restrict__ row2expanded,       // [R]
    int const* __restrict__ row2expert,         // [R] (-1 = padding row)
    __nv_bfloat16* __restrict__ out,            // [R, I]
    int64_t R,
    int64_t I,
    int64_t K) {
  int64_t const total = R * I;
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    int64_t const r = idx / I;
    int64_t const h = idx % I;
    int const e = row2expert[r];
    if (e < 0) {
      out[r * I + h] = __float2bfloat16(0.0f);  // deterministic padding (P1+ may relax)
      continue;
    }
    int64_t const t = row2token[r];
    int64_t const x = row2expanded[r];
    __nv_bfloat16 const* a = hidden + t * K;
    __nv_bfloat16 const* wg = w + ((int64_t)e * 2 * I + 2 * h) * K;      // col 2h
    __nv_bfloat16 const* wu = w + ((int64_t)e * 2 * I + 2 * h + 1) * K;  // col 2h+1
    float acc_g = 0.0f, acc_u = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
      float const av = __bfloat162float(a[k]);
      acc_g += av * __bfloat162float(wg[k]);
      acc_u += av * __bfloat162float(wu[k]);
    }
    float x1 = acc_g;  // interleaved col 2h
    float x2 = acc_u;  // interleaved col 2h+1
    if (delta != nullptr) {
      x1 += __bfloat162float(delta[x * 2 * I + I + h]);  // 2nd contiguous half
      x2 += __bfloat162float(delta[x * 2 * I + h]);      // 1st contiguous half
    }
    out[r * I + h] = __float2bfloat16(silu(x2) * x1);
  }
}

// P3: properly-gridded row gather (replaces moe::dev::permute on the bf16 fold path).
// The dev permute kernel launches a decode-shaped tiny grid (128 blocks, 11% occupancy at
// prefill shapes -> 180 us/layer for ~8 us of HBM traffic). This is a plain bandwidth-bound
// gather: out[r, :] = hidden[row2token[r], :] for valid rows, 16B vectorized, flat grid.
__global__ void gatherRowsKernel(
    __nv_bfloat16 const* __restrict__ hidden,  // [num_tokens, K]
    int const* __restrict__ row2token,         // [R] (-1 or OOB => skip row)
    __nv_bfloat16* __restrict__ out,           // [R, K]
    int64_t R,
    int64_t K,
    int64_t num_tokens) {
  int64_t const vecs_per_row = K / 8;  // 8 x bf16 = 16B
  int64_t const total = R * vecs_per_row;
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    int64_t const r = idx / vecs_per_row;
    int64_t const v = idx % vecs_per_row;
    int const t = row2token[r];
    if (t < 0 || t >= num_tokens) continue;  // padding rows stay garbage (base contract)
    reinterpret_cast<uint4*>(out + r * K)[v] =
        reinterpret_cast<uint4 const*>(hidden + (int64_t)t * K)[v];
  }
}

}  // namespace sgl_bf16_fold

// ---- FFI ----

// Probe: returns {CUTLASS_MAJOR, CUTLASS_MINOR, 1} — proves the cutlass include path works
// inside THIS module's build (P1 prerequisite).
Array<int64_t> sgl_bf16_fold_probe() {
  return {
      static_cast<int64_t>(cutlass::getVersionMajor()),
      static_cast<int64_t>(cutlass::getVersionMinor()),
      static_cast<int64_t>(1)};
}

void sgl_bf16_moe_gemm1_fold_ref(
    TensorView hidden,                       // [num_tokens, K] bf16
    TensorView w_fold,                       // [E, 2I, K] bf16
    Optional<TensorView> gate_up_lora_delta, // [num_expanded, 2I] bf16 (flattened) or absent
    TensorView permuted_row_to_token,        // [R] int32
    TensorView permuted_row_to_expanded,     // [R] int32
    TensorView permuted_row_to_expert,       // [R] int32 (-1 padding)
    TensorView activated_out) {              // [R, I] bf16
  TVM_FFI_ICHECK_EQ(hidden.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(w_fold.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(activated_out.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK(hidden.ndim() == 2 && w_fold.ndim() == 3 && activated_out.ndim() == 2);
  TVM_FFI_ICHECK(hidden.IsContiguous() && w_fold.IsContiguous() && activated_out.IsContiguous());
  int64_t const K = hidden.size(1);
  int64_t const I = activated_out.size(1);
  int64_t const R = activated_out.size(0);
  TVM_FFI_ICHECK_EQ(w_fold.size(1), 2 * I) << "w_fold must be [E, 2I, K] (interleaved g/u)";
  TVM_FFI_ICHECK_EQ(w_fold.size(2), K);
  TVM_FFI_ICHECK(
      permuted_row_to_token.numel() >= R && permuted_row_to_expanded.numel() >= R &&
      permuted_row_to_expert.numel() >= R);
  __nv_bfloat16 const* delta_ptr = nullptr;
  if (gate_up_lora_delta.has_value()) {
    TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().dtype(), dl_bfloat16);
    TVM_FFI_ICHECK(gate_up_lora_delta.value().IsContiguous());
    TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().numel() % (2 * I), 0);
    delta_ptr = static_cast<__nv_bfloat16 const*>(gate_up_lora_delta.value().data_ptr());
  }
  auto device = hidden.device();
  cudaStream_t stream = get_stream(device);
  int64_t const total = R * I;
  int const threads = 256;
  int const blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  sgl_bf16_fold::foldRefKernel<<<blocks, threads, 0, stream>>>(
      static_cast<__nv_bfloat16 const*>(hidden.data_ptr()),
      static_cast<__nv_bfloat16 const*>(w_fold.data_ptr()),
      delta_ptr,
      static_cast<int const*>(permuted_row_to_token.data_ptr()),
      static_cast<int const*>(permuted_row_to_expanded.data_ptr()),
      static_cast<int const*>(permuted_row_to_expert.data_ptr()),
      static_cast<__nv_bfloat16*>(activated_out.data_ptr()),
      R,
      I,
      K);
}

// P3 gather: out[r,:] = hidden[row2token[r],:] (K % 8 == 0; 16B vectorized).
void sgl_bf16_gather_rows(
    TensorView hidden,        // [num_tokens, K] bf16
    TensorView row2token,     // [R] int32
    TensorView out) {         // [R, K] bf16
  TVM_FFI_ICHECK_EQ(hidden.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(row2token.dtype(), dl_int32);
  TVM_FFI_ICHECK(hidden.ndim() == 2 && out.ndim() == 2);
  TVM_FFI_ICHECK(hidden.IsContiguous() && out.IsContiguous() && row2token.IsContiguous());
  int64_t const K = hidden.size(1);
  int64_t const R = out.size(0);
  TVM_FFI_ICHECK_EQ(out.size(1), K);
  TVM_FFI_ICHECK_EQ(K % 8, 0) << "K must be a multiple of 8 (16B vectorized gather).";
  TVM_FFI_ICHECK_EQ(row2token.numel(), R);
  auto device = hidden.device();
  cudaStream_t stream = get_stream(device);
  int64_t const total = R * (K / 8);
  int const threads = 256;
  int const blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  sgl_bf16_fold::gatherRowsKernel<<<blocks, threads, 0, stream>>>(
      static_cast<__nv_bfloat16 const*>(hidden.data_ptr()),
      static_cast<int const*>(row2token.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      R,
      K,
      hidden.size(0));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_bf16_fold_probe, sgl_bf16_fold_probe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_bf16_gather_rows, sgl_bf16_gather_rows);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_bf16_moe_gemm1_fold_ref, sgl_bf16_moe_gemm1_fold_ref);
