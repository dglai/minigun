#ifndef MINIGUN_CUDA_ADVANCE_CUH_
#define MINIGUN_CUDA_ADVANCE_CUH_

#include "./cuda_common.h"
#include "../advance.h"

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

template <typename VFrame,
          typename EFrame,
          typename Functor>
__global__ void CUDAAdvanceKernel(
    Csr csr,  // pass by value to make sure it copies the length to device memory
    VFrame* vframe,
    EFrame* eframe,
    IntArray1D input_frontier,
    IntArray1D output_frontier) {
  mg_int tid = blockIdx.x * blockDim.x + threadIdx.x;
  mg_int stride = blockDim.x * gridDim.x;
  mg_int eid = tid;
  while (eid < csr.column_indices.length) {
    src = BinarySearchSrc(csr.row_offsets, eid);
    dst = csr.column_indices[eid];
    if (Functor::CondEdge(src, dst, eid, vframe, eframe)) {
      Functor::ApplyEdge(src, dst, eid, vframe, eframe);
      // Add dst/eid to output frontier
    } else {
      // Add invalid to output frontier
    }
    eid += stride;
  }
};


template <typename VFrame,
          typename EFrame,
          typename Functor,
          typename Alloc>
struct DispatchXPU<kDLGPU, VFrame, EFrame, Functor, Alloc> {
  static void Advance(
      const RuntimeConfig& config,
      const Csr& csr,
      VFrame* vframe,
      EFrame* eframe,
      IntArray1D input_frontier,
      IntArray1D output_frontier,
      const Alloc& alloc) {
    CHECK(output_frontier->length != 0);
    int num_blocks = (csr.column_indices.length + 1023) / 1024;
    CUDA_CALL(CUDAAdvanceKernel<VFrame, EFrame, Functor>
      <<<num_blocks, 1024, 0, config.stream>>>(
        csr, vframe, eframe, input_frontier, output_frontier));
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_CUH_
