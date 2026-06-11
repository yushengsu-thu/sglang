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

// This TU is deliberately FREE of tvm-ffi headers: tvm_ffi_utils.h injects an unqualified
// `Tensor` into the global namespace which makes the unqualified `Tensor` inside CUTLASS
// epilogue headers ambiguous vs cute::Tensor at template-instantiation time. The FFI layer
// lives in bf16_moe_gemm1_grouped_ffi.cu and calls run_grouped() below.
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/util/packed_stride.hpp>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>

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

using ArchTag = cutlass::arch::Sm100;  // mainloop builder rejects Sm103 for dense ptr-array; sm_103a device guards mostly alias SM100 features
using OpClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<_256, _128, _64>;      // 2SM UMMA: cluster-wide MMA tile (128/SM)
using ClusterShape = Shape<_2, _1, _1>;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

// NOTE: ScheduleAuto does NOT pick the ptr-array mainloop for GroupProblemShape — the
// grouped (array-of-pointers) kernels must be requested explicitly via the PtrArray tags.
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    void, LayoutD*, AlignD,           // no C source (beta = 0)
    ElementD, LayoutD*, AlignD,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA*, AlignA,
    ElementB, LayoutB*, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100>::CollectiveOp;

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

// Plain C++ entry point (called from bf16_moe_gemm1_grouped_ffi.cu). Returns nullptr on
// success or a static error string. Owns its scratch/workspace via stream-ordered allocs.
char const* run_grouped(
    void const* permuted_hidden,
    void const* w_fold,
    int const* num_tokens_per_expert,
    int E,
    int N,
    int K,
    int tile,
    void* gate_up_out,
    cudaStream_t stream) {
  // Device scratch for per-group args (single stream-ordered alloc, 16B-aligned slots).
  auto align16 = [](size_t x) { return (x + 15) & ~size_t(15); };
  size_t off0 = 0;
  size_t const o_shapes = off0; off0 += align16(sizeof(UnderlyingShape) * E);
  size_t const o_pa = off0; off0 += align16(sizeof(void*) * E);
  size_t const o_pb = off0; off0 += align16(sizeof(void*) * E);
  size_t const o_pd = off0; off0 += align16(sizeof(void*) * E);
  size_t const o_sa = off0; off0 += align16(sizeof(StrideA) * E);
  size_t const o_sb = off0; off0 += align16(sizeof(StrideB) * E);
  size_t const o_sd = off0; off0 += align16(sizeof(StrideD) * E);
  char* base = nullptr;
  if (cudaMallocAsync(&base, off0, stream) != cudaSuccess) return "scratch alloc failed";

  auto* shapes = reinterpret_cast<UnderlyingShape*>(base + o_shapes);
  auto* ptrA = reinterpret_cast<ElementA const**>(base + o_pa);
  auto* ptrB = reinterpret_cast<ElementB const**>(base + o_pb);
  auto* ptrD = reinterpret_cast<ElementD**>(base + o_pd);
  auto* sA = reinterpret_cast<StrideA*>(base + o_sa);
  auto* sB = reinterpret_cast<StrideB*>(base + o_sb);
  auto* sD = reinterpret_cast<StrideD*>(base + o_sd);

  buildGroupArgsKernel<<<1, 32, 0, stream>>>(
      num_tokens_per_expert, E, tile, N, K,
      static_cast<ElementA const*>(permuted_hidden),
      static_cast<ElementB const*>(w_fold),
      static_cast<ElementD*>(gate_up_out),
      shapes, ptrA, ptrB, ptrD, sA, sB, sD);

  // Persistent ptr-array scheduler sizes its grid from hw_info.sm_count — leaving it
  // unset (0) makes run() fail with kErrorInternal before any CUDA call.
  cutlass::KernelHardwareInfo hw_info;
  cudaGetDevice(&hw_info.device_id);
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {E, shapes, /*host_problem_shapes=*/nullptr},
      {ptrA, sA, ptrB, sB},
      {{}, /*ptr_C=*/nullptr, sD, ptrD, sD},
      hw_info,
  };
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws = nullptr;
  if (ws_size > 0 && cudaMallocAsync(&ws, ws_size, stream) != cudaSuccess) {
    cudaFreeAsync(base, stream);
    return "workspace alloc failed";
  }
  static thread_local char errbuf[256];
  char const* err = nullptr;
  cudaError_t cerr = cudaGetLastError();
  if (cerr != cudaSuccess) {
    snprintf(errbuf, sizeof(errbuf), "pre-existing cuda error: %s", cudaGetErrorString(cerr));
    err = errbuf;
  }
  if (err == nullptr) {
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      snprintf(errbuf, sizeof(errbuf), "can_implement: %s", cutlass::cutlassGetStatusString(status));
      err = errbuf;
    } else {
      status = gemm.initialize(args, ws, stream);
      if (status != cutlass::Status::kSuccess) {
        snprintf(errbuf, sizeof(errbuf), "initialize: %s", cutlass::cutlassGetStatusString(status));
        err = errbuf;
      } else {
        status = gemm.run(stream);
        if (status != cutlass::Status::kSuccess) {
          cudaError_t lc = cudaGetLastError();
          snprintf(errbuf, sizeof(errbuf), "run: %s (cuda: %s)",
                   cutlass::cutlassGetStatusString(status), cudaGetErrorString(lc));
          err = errbuf;
        }
      }
    }
  }
  if (ws != nullptr) cudaFreeAsync(ws, stream);
  cudaFreeAsync(base, stream);
  return err;
}

}  // namespace sgl_bf16_fold_p1
