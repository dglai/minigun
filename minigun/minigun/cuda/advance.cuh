#ifndef MINIGUN_CUDA_ADVANCE_CUH_
#define MINIGUN_CUDA_ADVANCE_CUH_

#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>

#include "../advance.h"
#include "./cuda_common.h"
#include "./tuning.h"

namespace minigun {
namespace advance {

// Binary search the row_offsets to find the source node of the edge id.
__device__ __forceinline__ mg_int BinarySearchSrc(const IntArray1D& array, mg_int eid) {
  mg_int lo = 0, hi = array.length - 1;
  while (lo < hi) {
    mg_int mid = (lo + hi) >> 1;
    if (array.data[mid] < eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // INVARIANT: lo == hi
  if (array.data[hi] == eid) {
    return hi;
  } else {
    return hi - 1;
  }
}

template <typename Config,
          typename GData,
          typename Functor>
__global__ void CUDAAdvanceAllKernel(
    Csr csr,  // pass by value to make sure it is copied to device memory
    GData* gdata,
    IntArray1D output_frontier) {
  // TODO(minjie): load input_froniter subgraph into shared memory
  mg_int ty = blockIdx.y * blockDim.y + threadIdx.y;
  mg_int stride_y = blockDim.y * gridDim.y;
  mg_int eid = ty;
  while (eid < csr.column_indices.length) {
    mg_int src = BinarySearchSrc(csr.row_offsets, eid);
    mg_int dst = csr.column_indices.data[eid];
    if (Functor::CondEdge(src, dst, eid, gdata)) {
      Functor::ApplyEdge(src, dst, eid, gdata);
      // Add dst/eid to output frontier
    } else {
      // Add invalid to output frontier
    }
    eid += stride_y;
  }
};

/*
if advance all, input_frontier is not needed, output frontier is all.
else if len(input_frontier) == 0, do nothing.

if no_output_frontier, output length is not needed.
else {
  if output_frontier is none, compute
  else just use
}
*/

__inline__ void ComputeEdgeCounts(
    MgpuContext& mgpuctx,
    const Csr& csr,
    IntArray1D* edge_counts) {
}

__inline__ void ComputeOutputLength(
    MgpuContext& mgpuctx,
    const Csr& csr,
    const IntArray1D& input_frontier,
    const IntArray1D& edge_counts,
    mg_int* outlen) {
}

template <typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
struct DispatchXPU<kDLGPU, Config, GData, Functor, Alloc> {
  static void Advance(
      const RuntimeConfig& rtcfg,
      const Csr& csr,
      GData* gdata,
      IntArray1D input_frontier,
      IntArray1D output_frontier,
      Alloc alloc) {
    MgpuContext mgpuctx(rtcfg.stream);
    // An array of size len(input_frointer). Each element i is the number of
    // edges input_frointer[i] has.
    IntArray1D edge_counts;

    if (Config::kAdvanceAll) {
      // input_frontier and edge_counts will not be used.
      if (Config::kRequireOutFront && output_frontier.data == nullptr) {
        // Allocate output frointer buffer, the length is equal to the number
        // of edges.
        output_frontier.length = csr.column_indices.length;
        output_frontier.data = alloc.template AllocateData<mg_int>(
            output_frontier.length * sizeof(mg_int));
      }
    } else {
      edge_counts.length = input_frontier.length;
      edge_counts.data = alloc.template AllocateWorkspace<mg_int>(
            edge_counts.length * sizeof(mg_int));
      ComputeEdgeCounts(mgpuctx, csr, &edge_counts);
      if (Config::kRequireOutFront && output_frontier.data == nullptr) {
        // The output frontier buffer should be allocated.
        ComputeOutputLength(mgpuctx,
            csr, input_frontier, edge_counts, &output_frontier.length);
        output_frontier.data = alloc.template AllocateData<mg_int>(
            output_frontier.length * sizeof(mg_int));
      }
    }

    // Call advance
    CHECK_GT(rtcfg.data_num_blocks, 0);
    CHECK_GT(rtcfg.data_num_threads, 0);
    KernelConfig kcfg;
    TuneKernelConfig<Config>(rtcfg, csr, input_frontier, output_frontier, &kcfg);
    const dim3 nblks(rtcfg.data_num_blocks, kcfg.by);
    const dim3 nthrs(rtcfg.data_num_threads, kcfg.ty);
    LOG(INFO) << "Blocks: (" << nblks.x << "," << nblks.y << ") Threads: ("
      << nthrs.x << "," << nthrs.y << ")";

    if (Config::kAdvanceAll) {
      CUDAAdvanceAllKernel<Config, GData, Functor>
        <<<nblks, nthrs, 0, rtcfg.stream>>>(csr, gdata, output_frontier);
    } else {
      LOG(FATAL) << "!!!";
    }

    if (edge_counts.data) {
      alloc.FreeWorkspace(edge_counts.data);
    }
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_CUH_
