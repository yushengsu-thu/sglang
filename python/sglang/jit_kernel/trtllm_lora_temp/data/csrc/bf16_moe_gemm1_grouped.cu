// opt7 P1 — bf16 grouped GEMM for the in-MoE fold (plain epilogue).
//
// Computes, per local expert e:  D_e[M_e, N] = A_e[M_e, K] x W_e[N, K]^T   (bf16, fp32 accum)
// where A is the (already-)permuted hidden buffer with tile-padded per-expert segments and
// W is the plain row-major fold weight [E, N=2I, K] (interleaved g/u columns).
//
// P1 scope (see opt7_design/OPT7_DESIGN.md):
//   * A = pre-permuted buffer (the existing moe::dev::permute keeps running) — isolates GEMM
//     perf for the parity gate vs the tuned bmm_Bfloat16 cubin (~57 us @ 4096-tok shapes).
//   * Plain epilogue (raw gate_up out). P2 swaps in the SwiGLU+LoRA EVT; P3 adds the gather.
//   * Per-expert segment layout: offset_e = sum_{j<e} round_up(cnt_j, tile) — the same
//     tile-padded grouping trtllm routing produces. The P1 unit/bench driver constructs A in
//     this convention; integration fidelity (cta_idx maps) is P4.
//
// CUTLASS 4.x Sm100 ptr-array (grouped) GEMM via CollectiveBuilder; problem shapes, per-group
// pointers and strides are built on-device from num_tokens_per_expert (one tiny kernel).

#include "tvm_ffi_utils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace sgl_bf16_fold_p1 {

using namespace cute;

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;

// A: [M_e, K] row-major; B: W_e [N, K] row-major consumed as [K, N] column-major (TN GEMM);
// D: [M_e, N] row-major.
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignA = 8, AlignB = 8, AlignD = 8;

using ArchTag = cutlass::arch::Sm100;
using OpClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<_128, _128, _64>;      // first cut; tuned later against the parity gate
using ClusterShape = Shape<_1, _1, _1>;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    void, LayoutD*, AlignD,           // no C source (beta = 0)
    ElementD, LayoutD*, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA*, AlignA,
    ElementB, LayoutB*, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using UnderlyingShape = typename ProblemShape::UnderlyingProblemShape;

// Build per-expert grouped-GEMM arguments on device from the routing counts.
// Single block; thread 0 walks the (tiny, E<=64) prefix sum of tile-padded counts.
__global__ void buildGroupArgsKernel(
    int const* __restrict__ num_tokens_per_expert,  // [E] real counts
    int E,
    int tile,
    int N,
    int K,
    ElementA const* A_base,   // [R_padded, K]
    ElementB const* B_base,   // [E, N, K]
    ElementD* D_base,         // [R_padded, N]
    UnderlyingShape* shapes,
    ElementA const** ptrA,
    ElementB const** ptrB,
    ElementD** ptrD,
    StrideA* sA,
    StrideB* sB,
    StrideD* sD) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int64_t off = 0;
  for (int e = 0; e < E; ++e) {
    int const cnt = num_tokens_per_expert[e];
    int const m = ((cnt + tile - 1) / tile) * tile;  // tile-padded segment (pad rows computed; harmless)
    shapes[e] = UnderlyingShape{m, N, K};
    ptrA[e] = A_base + off * K;
    ptrB[e] = B_base + (int64_t)e * N * K;
    ptrD[e] = D_base + off * N;
    sA[e] = cutlass::make_cute_packed_stride(StrideA{}, {m, K, 1});
    sB[e] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    sD[e] = cutlass::make_cute_packed_stride(StrideD{}, {m, N, 1});
    off += m;
  }
}

}  // namespace sgl_bf16_fold_p1

// FFI: grouped gate_up GEMM over tile-padded permuted A.
//   permuted_hidden [R_padded, K] bf16, w_fold [E, N, K] bf16, num_tokens_per_expert [E] i32,
//   gate_up_out [R_padded, N] bf16 (written; padding rows = garbage, same as today's contract).
void sgl_bf16_moe_gemm1_grouped(
    TensorView permuted_hidden,
    TensorView w_fold,
    TensorView num_tokens_per_expert,
    int64_t tile,
    TensorView gate_up_out) {
  namespace p1 = sgl_bf16_fold_p1;
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

  auto device = permuted_hidden.device();
  cudaStream_t stream = get_stream(device);

  // Device scratch for per-group args (alignment-safe single buffer).
  size_t const bytes_shapes = sizeof(p1::UnderlyingShape) * E;
  size_t const bytes_ptr = sizeof(void*) * E;
  size_t const bytes_sA = sizeof(p1::StrideA) * E;
  size_t const bytes_sB = sizeof(p1::StrideB) * E;
  size_t const bytes_sD = sizeof(p1::StrideD) * E;
  auto align16 = [](size_t x) { return (x + 15) & ~size_t(15); };
  size_t off0 = 0;
  size_t const o_shapes = off0; off0 += align16(bytes_shapes);
  size_t const o_pa = off0; off0 += align16(bytes_ptr);
  size_t const o_pb = off0; off0 += align16(bytes_ptr);
  size_t const o_pd = off0; off0 += align16(bytes_ptr);
  size_t const o_sa = off0; off0 += align16(bytes_sA);
  size_t const o_sb = off0; off0 += align16(bytes_sB);
  size_t const o_sd = off0; off0 += align16(bytes_sD);
  Tensor scratch = alloc_tensor({(int64_t)off0}, dl_int8, device);
  char* base = static_cast<char*>(scratch.data_ptr());

  auto* shapes = reinterpret_cast<p1::UnderlyingShape*>(base + o_shapes);
  auto* ptrA = reinterpret_cast<p1::ElementA const**>(base + o_pa);
  auto* ptrB = reinterpret_cast<p1::ElementB const**>(base + o_pb);
  auto* ptrD = reinterpret_cast<p1::ElementD**>(base + o_pd);
  auto* sA = reinterpret_cast<p1::StrideA*>(base + o_sa);
  auto* sB = reinterpret_cast<p1::StrideB*>(base + o_sb);
  auto* sD = reinterpret_cast<p1::StrideD*>(base + o_sd);

  p1::buildGroupArgsKernel<<<1, 32, 0, stream>>>(
      static_cast<int const*>(num_tokens_per_expert.data_ptr()),
      E,
      static_cast<int>(tile),
      N,
      K,
      static_cast<p1::ElementA const*>(permuted_hidden.data_ptr()),
      static_cast<p1::ElementB const*>(w_fold.data_ptr()),
      static_cast<p1::ElementD*>(gate_up_out.data_ptr()),
      shapes,
      ptrA,
      ptrB,
      ptrD,
      sA,
      sB,
      sD);

  typename p1::Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {E, shapes, /*host_problem_shapes=*/nullptr},
      {ptrA, sA, ptrB, sB},
      {{}, /*ptr_C=*/nullptr, sD, ptrD, sD},
  };
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;

  p1::Gemm gemm;
  size_t ws_size = p1::Gemm::get_workspace_size(args);
  Tensor ws = alloc_tensor({(int64_t)std::max<size_t>(ws_size, 16)}, dl_int8, device);
  auto status = gemm.can_implement(args);
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
      << "bf16 grouped GEMM can_implement failed: " << cutlass::cutlassGetStatusString(status);
  status = gemm.initialize(args, ws.data_ptr(), stream);
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
      << "bf16 grouped GEMM initialize failed: " << cutlass::cutlassGetStatusString(status);
  status = gemm.run(stream);
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
      << "bf16 grouped GEMM run failed: " << cutlass::cutlassGetStatusString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_bf16_moe_gemm1_grouped, sgl_bf16_moe_gemm1_grouped);
