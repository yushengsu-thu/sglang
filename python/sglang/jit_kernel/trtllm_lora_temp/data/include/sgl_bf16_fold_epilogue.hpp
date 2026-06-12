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
  using EpilogueTile = decltype(cute::take<0, 2>(CtaTileShapeMNK_{}));
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

    // Chunked fold: 8 accumulator elements (4 interleaved pairs -> 4 folded outputs) per
    // step. Within a chunk we verify n-contiguity + same row once, hoist the per-row
    // perm2exp lookup, vector-load the two half-contiguous delta runs (4x bf16 = 8B each)
    // and vector-store the 4 folded bf16 (8B). Misaligned chunks fall back to scalars.
    int last_m = -1, last_x = -1;
    CUTLASS_PRAGMA_UNROLL
    for (int u = 0; u < size(frag); u += 8) {
      auto c0 = coords(u);
      int const m = int(get<0>(c0));
      int const n0 = int(get<1>(c0));
      if (m != last_m) {
        last_m = m;
        last_x = (m < int(M)) ? perm2exp[m] : -1;
      }
      if (last_x < 0 || m >= int(M)) {
        continue;  // padding row or OOB: leave D garbage (same contract as base path)
      }
      auto c7 = coords(u + 7);
      bool const fast = (int(get<0>(c7)) == m) && (int(get<1>(c7)) == n0 + 7) &&
                        ((n0 & 1) == 0) && (n0 + 7 < int(N));
      int64_t const drow = int64_t(m) * ld;
      if (fast) {
        int const h0 = n0 >> 1;
        float x1v[4], x2v[4];
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          x1v[j] = float(frag(u + 2 * j));
          x2v[j] = float(frag(u + 2 * j + 1));
        }
        if (params.delta != nullptr) {
          int64_t const base = int64_t(last_x) * (2 * I);
          // two 8B vector loads: delta[x, h0..h0+3] (-> x2) and delta[x, I+h0..I+h0+3] (-> x1)
          uint2 const d_lo = *reinterpret_cast<uint2 const*>(&params.delta[base + h0]);
          uint2 const d_hi = *reinterpret_cast<uint2 const*>(&params.delta[base + I + h0]);
          ElementD const* dlo = reinterpret_cast<ElementD const*>(&d_lo);
          ElementD const* dhi = reinterpret_cast<ElementD const*>(&d_hi);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 4; ++j) {
            x2v[j] += float(dlo[j]);
            x1v[j] += float(dhi[j]);
          }
        }
        ElementD outv[4];
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          outv[j] = ElementD(fold_silu(x2v[j]) * x1v[j]);
        }
        *reinterpret_cast<uint2*>(&ptr_D[drow + h0]) = *reinterpret_cast<uint2 const*>(outv);
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 8; j += 2) {
          auto cj = coords(u + j);
          int const mj = int(get<0>(cj));
          int const nj = int(get<1>(cj));
          if (mj >= int(M) || nj + 1 >= int(N) || (nj & 1)) continue;
          int const xj = (mj == m) ? last_x : perm2exp[mj];
          if (xj < 0) continue;
          int const h = nj >> 1;
          float x1 = float(frag(u + j));
          float x2 = float(frag(u + j + 1));
          if (params.delta != nullptr) {
            int64_t const base = int64_t(xj) * (2 * I);
            x1 += float(params.delta[base + I + h]);
            x2 += float(params.delta[base + h]);
          }
          ptr_D[int64_t(mj) * ld + h] = ElementD(fold_silu(x2) * x1);
        }
      }
    }

    return cute::make_tuple(acc_pipe_consumer_state, load_pipe_consumer_state);
  }

 private:
  Params params;
};

}  // namespace sgl_bf16_fold
