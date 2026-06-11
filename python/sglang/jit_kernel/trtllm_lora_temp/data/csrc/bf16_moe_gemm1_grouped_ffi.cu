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

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_bf16_moe_gemm1_grouped, sgl_bf16_moe_gemm1_grouped);
