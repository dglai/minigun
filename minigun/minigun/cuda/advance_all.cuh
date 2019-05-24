#ifndef MINIGUN_CUDA_ADVANCE_ALL_CUH_
#define MINIGUN_CUDA_ADVANCE_ALL_CUH_

#include "./cuda_common.cuh"

namespace minigun {
namespace advance {

#define MAX_NTHREADS 1024
#define PER_THREAD_WORKLOAD 1
#define MAX_NBLOCKS 65535

// Binary search the row_offsets to find the source node of the edge id.
template <typename Idx>
__device__ __forceinline__ Idx BinarySearchSrc(const IntArray1D<Idx>& array, Idx eid) {
  Idx lo = 0, hi = array.length - 1;
  while (lo < hi) {
    Idx mid = (lo + hi) >> 1;
    if (_ldg(array.data + mid) <= eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // INVARIANT: lo == hi
  if (_ldg(array.data + hi) == eid) {
    return hi;
  } else {
    return hi - 1;
  }
}

template <typename Idx,
          typename Config,
          typename GData,
          typename Functor>
__global__ void CudaAdvanceAllGunrockLBOutKernel(
    Csr<Idx> csr,
    GData gdata,
    IntArray1D<Idx> output_frontier) {
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  Idx stride_y = blockDim.y * gridDim.y;
  Idx eid = ty;
  while (eid < csr.column_indices.length) {
    // TODO(minjie): this is pretty inefficient; binary search is needed only
    //   when the thread is processing the neighbor list of a new node.
    Idx src = BinarySearchSrc(csr.row_offsets, eid);
    Idx dst = _ldg(csr.column_indices.data + eid);
    if (Functor::CondEdge(src, dst, eid, &gdata)) {
      Functor::ApplyEdge(src, dst, eid, &gdata);
      // Add dst/eid to output frontier
      if (Config::kMode == kV2V || Config::kMode == kE2V) {
        output_frontier.data[eid] = dst;
      } else if (Config::kMode == kV2E || Config::kMode == kE2E) {
        output_frontier.data[eid] = eid;
      }
    } else {
      if (Config::kMode != kV2N && Config::kMode != kE2N) {
        // Add invalid to output frontier
        output_frontier.data[eid] = MG_INVALID;
      }
    }
    eid += stride_y;
  }
};

template <typename Idx,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAllGunrockLBOut(
    const RuntimeConfig& rtcfg,
    const Csr<Idx>& csr,
    GData* gdata,
    IntArray1D<Idx> output_frontier,
    Alloc* alloc) {
  CHECK_GT(rtcfg.data_num_blocks, 0);
  CHECK_GT(rtcfg.data_num_threads, 0);
  const Idx M = csr.column_indices.length;
  const int ty = MAX_NTHREADS / rtcfg.data_num_threads;
  const int ny = ty * PER_THREAD_WORKLOAD;
  const int by = std::min((M + ny - 1) / ny, static_cast<Idx>(MAX_NBLOCKS));
  const dim3 nblks(rtcfg.data_num_blocks, by);
  const dim3 nthrs(rtcfg.data_num_threads, ty);
  //LOG(INFO) << "Blocks: (" << nblks.x << "," << nblks.y << ") Threads: ("
    //<< nthrs.x << "," << nthrs.y << ")";
  CudaAdvanceAllGunrockLBOutKernel<Idx, Config, GData, Functor>
    <<<nblks, nthrs, 0, rtcfg.stream>>>(csr, *gdata, output_frontier);
}

template <typename Idx,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAll(
    AdvanceAlg algo,
    const RuntimeConfig& rtcfg,
    const Csr<Idx>& csr,
    GData* gdata,
    IntArray1D<Idx>* output_frontier,
    Alloc* alloc) {
  Idx out_len = csr.column_indices.length;
  if (output_frontier) {
    if (output_frontier->data == nullptr) {
      // Allocate output frointer buffer, the length is equal to the number
      // of edges.
      output_frontier->length = out_len;
      output_frontier->data = alloc->template AllocateData<Idx>(
          output_frontier->length * sizeof(Idx));
    } else {
      CHECK_GE(output_frontier->length, out_len)
        << "Require output frontier of length " << out_len
        << " but only got a buffer of length " << output_frontier->length;
    }
  }
  IntArray1D<Idx> outbuf = (output_frontier)? *output_frontier : IntArray1D<Idx>();
  switch (algo) {
    case kGunrockLBOut :
      CudaAdvanceAllGunrockLBOut<Idx, Config, GData, Functor, Alloc>(
          rtcfg, csr, gdata, outbuf, alloc);
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
