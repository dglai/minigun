#ifndef MINIGUN_CUDA_ADVANCE_ALL_CUH_
#define MINIGUN_CUDA_ADVANCE_ALL_CUH_

#include "./cuda_common.cuh"
#include <algorithm>
#include <cstdio>

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
          typename DType,
          typename Config,
          typename GData,
          typename Functor>
__global__ void CudaAdvanceAllEdgeParallelCSRKernel(
    Csr<Idx> csr,
    GData gdata,
    bool is_out_csr) {
  const Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  Idx eid = ty;
  while (eid < csr.column_indices.length) {
    // NOTE(minjie): this is pretty inefficient; binary search is needed only
    //   when the thread is processing the neighbor list of a new node.
    const Idx src = BinarySearchSrc(csr.row_offsets, eid);
    const Idx dst = _ldg(csr.column_indices.data + eid);
    if (is_out_csr)
      Functor::ApplyEdge(src, dst, eid, &gdata);
    else
      Functor::ApplyEdge(dst, src, eid, &gdata);
    eid += stride_y;
  }
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAllEdgeParallelCSR(
    const RuntimeConfig& rtcfg,
    const Csr<Idx>& csr,
    GData* gdata,
    Alloc* alloc,
    bool is_out_csr) {
  CHECK_GT(rtcfg.data_num_blocks, 0);
  CHECK_GT(rtcfg.data_num_threads, 0);
  const Idx M = csr.column_indices.length;
  const int ty = 1; // MAX_NTHREADS / rtcfg.data_num_threads;
  const int ny = ty * PER_THREAD_WORKLOAD;
  const int by = std::min((M + ny - 1) / ny, static_cast<Idx>(MAX_NBLOCKS));
  const dim3 nblks(rtcfg.data_num_blocks, by);
  const dim3 nthrs(rtcfg.data_num_threads, ty);
  CudaAdvanceAllEdgeParallelCSRKernel<Idx, DType, Config, GData, Functor>
    <<<nblks, nthrs, 0, rtcfg.stream>>>(csr, *gdata, is_out_csr);
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor>
__global__ void CudaAdvanceAllEdgeParallelKernel(
    Coo<Idx> coo,
    GData gdata) {
  const Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  Idx eid = ty;
  while (eid < coo.column.length) {
    const Idx src = _ldg(coo.row.data + eid);
    const Idx dst = _ldg(coo.column.data + eid);
    Functor::ApplyEdge(src, dst, eid, &gdata);
    eid += stride_y;
  }
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAllEdgeParallel(
    const RuntimeConfig& rtcfg,
    const Coo<Idx>& coo,
    GData* gdata,
    Alloc* alloc) {
  CHECK_GT(rtcfg.data_num_blocks, 0);
  CHECK_GT(rtcfg.data_num_threads, 0);
  const Idx M = coo.column.length;
  const int ty = 1; // MAX_NTHREADS / rtcfg.data_num_threads;
  const int ny = ty * PER_THREAD_WORKLOAD;
  const int by = std::min((M + ny - 1) / ny, static_cast<Idx>(MAX_NBLOCKS));
  const dim3 nblks(rtcfg.data_num_blocks, by);
  const dim3 nthrs(rtcfg.data_num_threads, ty);
  CudaAdvanceAllEdgeParallelKernel<Idx, DType, Config, GData, Functor>
    <<<nblks, nthrs, 0, rtcfg.stream>>>(coo, *gdata);
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor>
__global__ void CudaAdvanceAllNodeParallelKernel(
    Csr<Idx> csr,
    GData gdata) {
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
  Idx stride_y = blockDim.y * gridDim.y;
  Idx stride_x = blockDim.x * gridDim.x;
  Idx feat_size = Functor::GetFeatSize(&gdata);
  DType *outbuf = Functor::GetOutBuf(&gdata);
  DType val;
  Idx vid = ty;
  while (vid < csr.row_offsets.length - 1) {
    Idx start = _ldg(csr.row_offsets.data + vid);
    Idx end = _ldg(csr.row_offsets.data + vid + 1);
    Idx feat_idx = tx;
    if (start < end) {
      while (feat_idx < feat_size) {
        Idx outoff = Functor::GetOutOffset(vid, &gdata) * feat_size + feat_idx;
        if (outbuf != nullptr)
          val = _ldg(outbuf + outoff);
        for (Idx eid = start; eid < end; ++eid) {
          Idx src, dst;
          if (Config::kParallel == kDst) {
            src = _ldg(csr.column_indices.data + eid);
            dst = vid;
          } else { // kSrc
            dst = _ldg(csr.column_indices.data + eid);
            src = vid;
          }
          Functor::ApplyEdgeReduce(src, dst, eid, feat_idx, &val, &gdata);
        }
        if (outbuf != nullptr)
          outbuf[outoff] = val;
        feat_idx += stride_x;
      }
    }
    vid += stride_y;
  }
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAllNodeParallel(
    const RuntimeConfig& rtcfg,
    const Csr<Idx>& csr,
    GData* gdata,
    Alloc* alloc) {
  CHECK_GT(rtcfg.data_num_blocks, 0);
  CHECK_GT(rtcfg.data_num_threads, 0);
  const Idx N = csr.row_offsets.length - 1;
  const int ty = 1; //MAX_NTHREADS / rtcfg.data_num_threads;
  const int ny = ty * PER_THREAD_WORKLOAD;
  const int by = std::min((N + ny - 1) / ny, static_cast<Idx>(MAX_NBLOCKS));
  const dim3 nblks(rtcfg.data_num_blocks, by);
  const dim3 nthrs(rtcfg.data_num_threads, ty);
  CudaAdvanceAllNodeParallelKernel<Idx, DType, Config, GData, Functor>
    <<<nblks, nthrs, 0, rtcfg.stream>>>(csr, *gdata);
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAll(
    const RuntimeConfig& rtcfg,
    const SpMat<Idx> &spmat,
    GData* gdata,
    Alloc* alloc) {
  switch (Config::kParallel) {
    case kSrc:
      if (spmat.out_csr != nullptr)
        CudaAdvanceAllNodeParallel<Idx, DType, Config, GData, Functor, Alloc>(
            rtcfg, *spmat.out_csr, gdata, alloc);
      else
        LOG(FATAL) << "out_csr must be created in source parallel mode";
      break;
    case kEdge:
      if (spmat.coo != nullptr)
        CudaAdvanceAllEdgeParallel<Idx, DType, Config, GData, Functor, Alloc>(
            rtcfg, *spmat.coo, gdata, alloc);
      else if (spmat.out_csr != nullptr)
        CudaAdvanceAllEdgeParallelCSR<Idx, DType, Config, GData, Functor, Alloc>(
            rtcfg, *spmat.out_csr, gdata, alloc, true);
      else if (spmat.in_csr != nullptr)
        CudaAdvanceAllEdgeParallelCSR<Idx, DType, Config, GData, Functor, Alloc>(
            rtcfg, *spmat.in_csr, gdata, alloc, false);
      else
        LOG(FATAL) << "At least one sparse format should be created.";
      break;
    case kDst:
      if (spmat.in_csr != nullptr)
        CudaAdvanceAllNodeParallel<Idx, DType, Config, GData, Functor, Alloc>(
            rtcfg, *spmat.in_csr, gdata, alloc);
      else
        LOG(FATAL) << "in_csr must be created in destination parallel mode.";
      break;
  }
}

#undef MAX_NTHREADS
#undef PER_THREAD_WORKLOAD
#undef MAX_NBLOCKS

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_ALL_CUH_
