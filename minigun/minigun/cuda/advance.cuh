#ifndef MINIGUN_CUDA_ADVANCE_CUH_
#define MINIGUN_CUDA_ADVANCE_CUH_

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

template <typename GData,
          typename Functor>
__global__ void CUDAAdvanceKernel(
    Csr csr,  // pass by value to make sure it is copied to device memory
    GData* gdata,
    IntArray1D input_frontier,
    IntArray1D output_frontier) {
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

template <typename GData,
          typename Functor,
          typename Alloc>
struct DispatchXPU<kDLGPU, GData, Functor, Alloc> {
  static void Advance(
      const RuntimeConfig& rtcfg,
      const Csr& csr,
      GData* gdata,
      IntArray1D input_frontier,
      IntArray1D output_frontier,
      const Alloc& alloc) {
    //CHECK(output_frontier.length != 0);
    CHECK_GT(rtcfg.data_num_blocks, 0);
    CHECK_GT(rtcfg.data_num_threads, 0);
    KernelConfig kcfg;
    TuneKernelConfig(rtcfg, csr, input_frontier, output_frontier, &kcfg);
    dim3 nblks(rtcfg.data_num_blocks, kcfg.by);
    dim3 nthrs(rtcfg.data_num_threads, kcfg.ty);
    LOG(INFO) << "Blocks: (" << nblks.x << "," << nblks.y << ") Threads: ("
      << nthrs.x << "," << nthrs.y << ")";
    CUDAAdvanceKernel<GData, Functor>
      <<<nblks, nthrs, 0, rtcfg.stream>>>(
        csr, gdata, input_frontier, output_frontier);
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_CUH_
