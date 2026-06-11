// opt7 P1 — FFI wrapper for the bf16 grouped gate_up GEMM. Kept in a separate TU from the
// CUTLASS instantiation: tvm_ffi_utils.h injects an unqualified global `Tensor` that makes
// CUTLASS epilogue headers ambiguous (vs cute::Tensor) at template-instantiation time.

#include "tvm_ffi_utils.h"

#include <cuda_runtime.h>

namespace sgl_bf16_fold_p1 {
char const* run_grouped(
    void const* permuted_hidden,
    void const* w_fold,
    int const* num_tokens_per_expert,
    int E,
    int N,
    int K,
    int tile,
    void* gate_up_out,
    cudaStream_t stream);
char const* run_grouped_fold(
    void const* permuted_hidden,
    void const* w_fold,
    int const* num_tokens_per_expert,
    int const* perm2exp,
    void const* delta,
    int E,
    int N,
    int K,
    int tile,
    void* activated_out,
    cudaStream_t stream);
}

// Grouped gate_up GEMM over tile-padded permuted A (see bf16_moe_gemm1_grouped.cu).
//   permuted_hidden [R_padded, K] bf16, w_fold [E, N, K] bf16, num_tokens_per_expert [E] i32,
//   gate_up_out [R_padded, N] bf16 (padding rows = garbage, same contract as today).
void sgl_bf16_moe_gemm1_grouped(
    TensorView permuted_hidden,
    TensorView w_fold,
    TensorView num_tokens_per_expert,
    int64_t tile,
    TensorView gate_up_out) {
  TVM_FFI_ICHECK_EQ(permuted_hidden.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(w_fold.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(gate_up_out.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(num_tokens_per_expert.dtype(), dl_int32);
  TVM_FFI_ICHECK(permuted_hidden.ndim() == 2 && w_fold.ndim() == 3 && gate_up_out.ndim() == 2);
  TVM_FFI_ICHECK(permuted_hidden.IsContiguous() && w_fold.IsContiguous() && gate_up_out.IsContiguous());
  int const E = static_cast<int>(w_fold.size(0));
  int const N = static_cast<int>(w_fold.size(1));
  int const K = static_cast<int>(w_fold.size(2));
  TVM_FFI_ICHECK_EQ(permuted_hidden.size(1), K);
  TVM_FFI_ICHECK_EQ(gate_up_out.size(1), N);
  TVM_FFI_ICHECK_EQ(gate_up_out.size(0), permuted_hidden.size(0));
  TVM_FFI_ICHECK_EQ(num_tokens_per_expert.numel(), E);
  TVM_FFI_ICHECK(N % 8 == 0 && K % 8 == 0) << "N/K must satisfy bf16 TMA alignment (8).";

  auto device = permuted_hidden.device();
  cudaStream_t stream = get_stream(device);
  char const* err = sgl_bf16_fold_p1::run_grouped(
      permuted_hidden.data_ptr(),
      w_fold.data_ptr(),
      static_cast<int const*>(num_tokens_per_expert.data_ptr()),
      E,
      N,
      K,
      static_cast<int>(tile),
      gate_up_out.data_ptr(),
      stream);
  TVM_FFI_ICHECK(err == nullptr) << "bf16 grouped GEMM failed: " << err;
}

// P2: fold-epilogue variant — writes the HALF-width activated output directly:
//   activated[r, h] = silu(acc[2h+1] + delta[x, h]) * (acc[2h] + delta[x, I+h]),
//   x = perm2exp[r] (-1 padding rows skipped). delta optional.
void sgl_bf16_moe_gemm1_fold_gemm(
    TensorView permuted_hidden,          // [R_padded, K] bf16
    TensorView w_fold,                   // [E, N=2I, K] bf16
    TensorView num_tokens_per_expert,    // [E] int32
    TensorView perm2exp,                 // [R_padded] int32
    Optional<TensorView> delta,          // [num_expanded, 2I] bf16
    int64_t tile,
    TensorView activated_out) {          // [R_padded, I] bf16
  TVM_FFI_ICHECK_EQ(permuted_hidden.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(w_fold.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(activated_out.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(num_tokens_per_expert.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(perm2exp.dtype(), dl_int32);
  int const E = static_cast<int>(w_fold.size(0));
  int const N = static_cast<int>(w_fold.size(1));
  int const K = static_cast<int>(w_fold.size(2));
  TVM_FFI_ICHECK_EQ(N % 2, 0);
  TVM_FFI_ICHECK_EQ(permuted_hidden.size(1), K);
  TVM_FFI_ICHECK_EQ(activated_out.size(1), N / 2);
  TVM_FFI_ICHECK_EQ(activated_out.size(0), permuted_hidden.size(0));
  TVM_FFI_ICHECK_EQ(perm2exp.numel(), permuted_hidden.size(0));
  void const* delta_ptr = nullptr;
  if (delta.has_value()) {
    TVM_FFI_ICHECK_EQ(delta.value().dtype(), dl_bfloat16);
    TVM_FFI_ICHECK(delta.value().IsContiguous());
    TVM_FFI_ICHECK_EQ(delta.value().numel() % N, 0);
    delta_ptr = delta.value().data_ptr();
  }
  auto device = permuted_hidden.device();
  cudaStream_t stream = get_stream(device);
  char const* err = sgl_bf16_fold_p1::run_grouped_fold(
      permuted_hidden.data_ptr(),
      w_fold.data_ptr(),
      static_cast<int const*>(num_tokens_per_expert.data_ptr()),
      static_cast<int const*>(perm2exp.data_ptr()),
      delta_ptr,
      E, N, K, static_cast<int>(tile),
      activated_out.data_ptr(),
      stream);
  TVM_FFI_ICHECK(err == nullptr) << "bf16 fold GEMM failed: " << err;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_bf16_moe_gemm1_grouped, sgl_bf16_moe_gemm1_grouped);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_bf16_moe_gemm1_fold_gemm, sgl_bf16_moe_gemm1_fold_gemm);
