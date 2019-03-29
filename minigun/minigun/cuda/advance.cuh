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
  /*
  for (mg_int l = 0; l < array.length - 1; ++l) {
    if (array.data[l] <= eid && array.data[l+1] > eid) {
      return l;
    }
  }*/
}

template <typename VFrame,
          typename EFrame,
          typename Functor>
__global__ void CUDAAdvanceKernel(
    Csr csr,  // pass by value to make sure it is copied to device memory
    VFrame vframe,
    EFrame eframe,
    IntArray1D input_frontier,
    IntArray1D output_frontier) {
  mg_int tid = blockIdx.x * blockDim.x + threadIdx.x;
  mg_int stride = blockDim.x * gridDim.x;
  mg_int eid = tid;
  while (eid < csr.column_indices.length) {
    mg_int src = BinarySearchSrc(csr.row_offsets, eid);
    mg_int dst = csr.column_indices.data[eid];
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
      VFrame vframe,
      EFrame eframe,
      IntArray1D input_frontier,
      IntArray1D output_frontier,
      const Alloc& alloc) {
    //CHECK(output_frontier.length != 0);
    int NUM_THREADS = 1024;
    int num_blocks = (csr.column_indices.length + NUM_THREADS-1) / NUM_THREADS;
    LOG(INFO) << "num_blocks: " << num_blocks;
    CUDAAdvanceKernel<VFrame, EFrame, Functor>
      <<<num_blocks, NUM_THREADS, 0, config.stream>>>(
        csr, vframe, eframe, input_frontier, output_frontier);
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_CUH_
