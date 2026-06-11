// opt7 P2 — SwiGLU+LoRA fold epilogue for the bf16 grouped gate_up GEMM.
//
// Fork of cutlass/epilogue/collective/sm100_epilogue_array_nosmem.hpp (CUTLASS 4.5,
// Sm100PtrArrayNoSmem specialization), with the elementwise store replaced by the in-MoE
// fold: the accumulator tile holds INTERLEAVED gate/up columns (g0,u0,g1,u1,...) of the
// gate_up projection; this epilogue folds adjacent column pairs
//
//     x1 = acc[m, 2h]   (+ delta[x, I + h])     // LoRA delta: HALF-CONTIGUOUS layout
//     x2 = acc[m, 2h+1] (+ delta[x, h])
//     D[m, h] = silu(x2) * x1                   // D is HALF width: [M, I = N/2]
//
// where x = group_perm2exp[m] maps the group-local permuted row to its expanded index
// (-1 padding rows are skipped). The GEMM's logical problem shape stays (M, N, K); only
// D's stride arrays describe the half-width output. DispatchPolicy stays
// Sm100PtrArrayNoSmem so GemmUniversal treats this exactly like the stock NoSmem epilogue
// (no load warp, no TMA staging, no smem).
//
// Semantics pinned by dev/test_bf16_fold_ref.py against the P0 reference kernel.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace sgl_bf16_fold {

using namespace cute;

CUTLASS_DEVICE float fold_silu(float x) { return x / (1.0f + expf(-x)); }

template <
    class CtaTileShapeMNK_,   // (CTA_M, CTA_N, CTA_K) of the GEMM (full-width N)
    class ElementAccumulator_,  // float
    class ElementD_,            // bf16
    class StrideD_,             // per-group pointer-to-stride of the HALF-width D
    class CopyOpT2R_>           // TMEM_LOAD op (reuse the one the stock builder picks)
class Sm100BF16FoldArrayEpilogue {
 public:
  using DispatchPolicy = cutlass::epilogue::Sm100PtrArrayNoSmem;
  // Nominal thread op: satisfies GemmUniversalAdapter's type queries; the actual math is
  // the fold below (this op is never invoked).
  using ThreadEpilogueOp =
      cutlass::epilogue::thread::LinearCombination<ElementD_, 1, ElementAccumulator_, float>;
  using CtaTileShapeMNK = CtaTileShapeMNK_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = float;
  using ElementScalar = float;
  using ElementOutput = ElementD_;
  using ElementC = void;        // no source
  using StrideC = StrideD_;     // placeholder (never dereferenced)
  using InternalStrideC = cute::remove_pointer_t<StrideC>;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using InternalStrideD = cute::remove_pointer_t<StrideD>;
  using CopyOpT2R = CopyOpT2R_;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  constexpr static int ThreadCount = 128;
  constexpr static uint32_t TmaTransactionBytes = 0;

  struct SharedStorage {
    struct TensorStorage {};
    struct TensorMapStorage {};
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;

  // Host-side arguments == device-side params.
  struct Arguments {
    ElementD** ptr_D = nullptr;          // per-group HALF-width D base pointers
    StrideD dD{};                        // per-group strides of D ([M, I] row-major)
    int const** ptr_perm2exp = nullptr;  // per-group: group-local permuted row -> expanded idx (-1 pad)
    ElementD const* delta = nullptr;     // [num_expanded, 2I] half-contiguous LoRA delta (nullable)
    int inner_dim = 0;                   // I = N/2
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(
      ProblemShape const&, Arguments const&, int /*sm_count*/ = 0) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const&, Arguments const&, void*, cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  Sm100BF16FoldArrayEpilogue(Params const& params, SharedStorage&) : params(params) {}

  template <
      bool ReuseTmem = false,
      class LoadPipeline,
      class LoadPipelineState,
      class AccumulatorPipeline,
      class AccumulatorPipelineState,
      class ProblemShapeMNKL,
      class TileShapeMNK,
      class TileCoordMNKL,
      class AccEngine,
      class AccLayout>
  CUTLASS_DEVICE auto operator()(
      [[maybe_unused]] LoadPipeline load_pipeline,
      [[maybe_unused]] LoadPipelineState load_pipe_consumer_state,
      AccumulatorPipeline acc_pipeline,
      AccumulatorPipelineState acc_pipe_consumer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK cta_tile_shape_mnk,
      TileCoordMNKL cta_coord_mnkl,
      cute::Tensor<AccEngine, AccLayout> const& accumulators,  // (MMA,MMA_M,MMA_N) TMEM
      [[maybe_unused]] SharedStorage&) {
    using X = Underscore;
    static_assert(is_tmem<AccEngine>::value, "Accumulator must be TMEM resident.");

    auto [M, N, K, L] = problem_shape_mnkl;            // N = full (interleaved) width
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
    if (K > 0) {
      acc_pipeline.consumer_wait(acc_pipe_consumer_state);
    }

    // T2R partition over the FULL-width accumulator tile (identical to the stock epilogue).
    auto cta_tiler = take<0, 2>(cta_tile_shape_mnk);
    Tensor tAcc = accumulators(make_coord(_, _), _0{}, _0{});               // (CTA_M,CTA_N)
    auto tiled_t2r = make_tmem_copy(CopyOpT2R{}, tAcc);
    auto thread_idx = threadIdx.x % size(tiled_t2r);
    auto thread_t2r = tiled_t2r.get_slice(thread_idx);

    // Coordinate tensor in FULL-width (m, n, l) space for predication + fold indexing.
    auto problem_shape_mnl = append<3>(make_shape(M, N), Int<1>{});
    auto cta_coord_mnl = append<3>(make_shape(m_coord, n_coord), Int<0>{});
    Tensor coordD = make_identity_tensor(problem_shape_mnl);
    Tensor cD = local_tile(coordD, cta_tiler, cta_coord_mnl);               // (CTA_M,CTA_N)
    Tensor tTR_cD = thread_t2r.partition_D(cD);

    Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);
    Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tTR_cD));
    if (K > 0) {
      copy(tiled_t2r, tTR_tAcc, tTR_rAcc);
      cutlass::arch::fence_view_async_tmem_load();
      acc_pipeline.consumer_release(acc_pipe_consumer_state);
      ++acc_pipe_consumer_state;
    } else {
      fill(tTR_rAcc, ElementAccumulator(0));
    }

    // ---- the fold ----
    // Coalesce to expose N-contiguous runs; tmem 32dpXb loads give each thread contiguous
    // column runs whose length is a multiple of 2 (CTA_N and the value layout are even),
    // so interleaved (g,u) pairs never straddle threads or runs.
    Tensor frag = coalesce(tTR_rAcc);
    Tensor coords = coalesce(tTR_cD);

    ElementD* ptr_D = params.ptr_D[l_coord];
    auto dD = params.dD[l_coord];                       // InternalStrideD, e.g. (I, _1, _0)
    int const* perm2exp = params.ptr_perm2exp[l_coord];
    int const I = params.inner_dim;
    int64_t const ld = int64_t(cute::get<0>(dD));       // row stride of half-width D

    CUTLASS_PRAGMA_UNROLL
    for (int u = 0; u < size(frag) / 2; ++u) {
      auto c0 = coords(2 * u);                          // (m, n, l) of the even column
      int const m = int(get<0>(c0));
      int const n = int(get<1>(c0));
      if (m >= int(M) || n + 1 >= int(N) || (n & 1)) {
        continue;  // out of bounds, or run misaligned (defensive; should not happen)
      }
      int const h = n >> 1;
      float x1 = float(frag(2 * u));                    // col 2h
      float x2 = float(frag(2 * u + 1));                // col 2h+1
      int const x = perm2exp[m];
      if (x < 0) {
        continue;  // padding row: leave D garbage (same contract as the base path)
      }
      if (params.delta != nullptr) {
        int64_t const base = int64_t(x) * (2 * I);
        x1 += float(params.delta[base + I + h]);
        x2 += float(params.delta[base + h]);
      }
      ptr_D[int64_t(m) * ld + h] = ElementD(fold_silu(x2) * x1);
    }

    return cute::make_tuple(acc_pipe_consumer_state, load_pipe_consumer_state);
  }

 private:
  Params params;
};

}  // namespace sgl_bf16_fold
