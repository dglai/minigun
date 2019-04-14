#ifndef MINIGUN_CUDA_ADVANCE_ALL_CUH_
#define MINIGUN_CUDA_ADVANCE_ALL_CUH_

namespace minigun {
namespace advance {

#define MAX_NTHREADS 1024
#define PER_THREAD_WORKLOAD 1
#define MAX_NBLOCKS 65535L

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
__global__ void CudaAdvanceAllGunrockLBOutKernel(
    Csr csr,  // pass by value to make sure it is copied to device memory
    GData* gdata,
    IntArray1D output_frontier) {
  mg_int ty = blockIdx.y * blockDim.y + threadIdx.y;
  mg_int stride_y = blockDim.y * gridDim.y;
  mg_int eid = ty;
  while (eid < csr.column_indices.length) {
    // TODO(minjie): this is pretty inefficient; binary search is needed only
    //   when the thread is processing the neighbor list of a new node.
    mg_int src = BinarySearchSrc(csr.row_offsets, eid);
    mg_int dst = csr.column_indices.data[eid];
    if (Functor::CondEdge(src, dst, eid, gdata)) {
      Functor::ApplyEdge(src, dst, eid, gdata);
      // Add dst/eid to output frontier
      if (Config::kMode == kV2V || Config::kMode == kE2V) {
        output_frontier.data[eid] = dst;
      } else if (Config::kMode == kV2E || Config::kMode == kE2E) {
        output_frontier.data[eid] = eid;
      }
    } else {
      if (Config::kMode != kV2N && Config::kMode != kE2N) {
        // Add invalid to output frontier
        output_frontier.data[eid] = kInvalid;
      }
    }
    eid += stride_y;
  }
};

template <typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAllGunrockLBOut(
    const RuntimeConfig& rtcfg,
    const Csr& csr,
    GData* gdata,
    IntArray1D output_frontier,
    Alloc* alloc) {
  CHECK_GT(rtcfg.data_num_blocks, 0);
  CHECK_GT(rtcfg.data_num_threads, 0);
  const mg_int M = csr.column_indices.length;
  const int ty = MAX_NTHREADS / rtcfg.data_num_threads;
  const int ny = ty * PER_THREAD_WORKLOAD;
  const int by = std::min((M + ny - 1) / ny, MAX_NBLOCKS);
  const dim3 nblks(rtcfg.data_num_blocks, by);
  const dim3 nthrs(rtcfg.data_num_threads, ty);
  //LOG(INFO) << "Blocks: (" << nblks.x << "," << nblks.y << ") Threads: ("
  //  << nthrs.x << "," << nthrs.y << ")";
  CudaAdvanceAllGunrockLBOutKernel<Config, GData, Functor>
    <<<nblks, nthrs, 0, rtcfg.stream>>>(csr, gdata, output_frontier);
}

template <typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAll(
    AdvanceAlg algo,
    const RuntimeConfig& rtcfg,
    const Csr& csr,
    GData* gdata,
    IntArray1D output_frontier,
    Alloc* alloc) {
  if (Config::kMode != kV2N && Config::kMode != kE2N
      && output_frontier.data == nullptr) {
    // Allocate output frointer buffer, the length is equal to the number
    // of edges.
    output_frontier.length = csr.column_indices.length;
    output_frontier.data = alloc->template AllocateData<mg_int>(
        output_frontier.length * sizeof(mg_int));
  }
  switch (algo) {
    case kGunrockLBOut :
      CudaAdvanceAllGunrockLBOut<Config, GData, Functor, Alloc>(
          rtcfg, csr, gdata, output_frontier, alloc);
      break;
    default:
      LOG(FATAL) << "Algorithm " << algo << " is not supported.";
  }
}

#undef MAX_NTHREADS
#undef PER_THREAD_WORKLOAD
#undef MAX_NBLOCKS

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_ALL_CUH_
