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

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor>
__global__ void CudaAdvanceAllGunrockLBOutKernel(
    Coo<Idx> coo,
    GData gdata,
    IntArray1D<Idx> output_frontier) {
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  Idx stride_y = blockDim.y * gridDim.y;
  Idx eid = ty;
  while (eid < coo.column.length) {
    Idx src = _ldg(coo.row.data + eid);
    Idx dst = _ldg(coo.column.data + eid);
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
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAllGunrockLBOut(
    const RuntimeConfig& rtcfg,
    const Coo<Idx>& coo,
    GData* gdata,
    IntArray1D<Idx> output_frontier,
    Alloc* alloc) {
  CHECK_GT(rtcfg.data_num_blocks, 0);
  CHECK_GT(rtcfg.data_num_threads, 0);
  const Idx M = coo.column.length;
  const int ty = MAX_NTHREADS / rtcfg.data_num_threads;
  const int ny = ty * PER_THREAD_WORKLOAD;
  const int by = std::min((M + ny - 1) / ny, static_cast<Idx>(MAX_NBLOCKS));
  const dim3 nblks(rtcfg.data_num_blocks, by);
  const dim3 nthrs(rtcfg.data_num_threads, ty);
  /*
  LOG(INFO) << "Blocks: (" << nblks.x << "," << nblks.y << ") Threads: ("
    << nthrs.x << "," << nthrs.y << ")";
  */
  CudaAdvanceAllGunrockLBOutKernel<Idx, DType, Config, GData, Functor>
    <<<nblks, nthrs, 0, rtcfg.stream>>>(coo, *gdata, output_frontier);
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
  if (Config::kParallel == kDst) {
    Idx dst = ty;
    while (dst < csr.row_offsets.length - 1) {
      Idx start = _ldg(csr.row_offsets.data + dst);
      Idx end = _ldg(csr.row_offsets.data + dst + 1);
      Idx feat_idx = tx;
      while (feat_idx < feat_size) {
        Idx outoff = dst * feat_size + feat_idx;
        if (outbuf != nullptr)
          val = _ldg(outbuf + outoff);
        for (Idx eid = start; eid < end; ++eid) {
          Idx src = _ldg(csr.column_indices.data + eid);
          if (Functor::CondEdge(src, dst, eid, &gdata))
            Functor::ApplyEdgeReduce(src, dst, eid, feat_idx, val, &gdata);
        }
        if (outbuf != nullptr)
          outbuf[outoff] = val;
        feat_idx += stride_x;
      }
      dst += stride_y;
    }
  } else {
    Idx src = ty;
    while (src < csr.row_offsets.length - 1) {
      Idx start = _ldg(csr.row_offsets.data + src);
      Idx end = _ldg(csr.row_offsets.data + src + 1);
      Idx feat_idx = tx;
      while (feat_idx < feat_size) {
        Idx outoff = src * feat_size + feat_idx;
        if (outbuf != nullptr)
          val = _ldg(outbuf + outoff);
        for (Idx eid = start; eid < end; ++eid) {
          Idx dst = _ldg(csr.column_indices.data + eid);
          if (Functor::CondEdge(src, dst, eid, &gdata))
            Functor::ApplyEdgeReduce(src, dst, eid, feat_idx, val, &gdata);
        }
        if (outbuf != nullptr)
          outbuf[outoff] = val;
        feat_idx += stride_x;
      }
      src += stride_y;
    }
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
    IntArray1D<Idx> output_frontier,
    Alloc* alloc) {
  CHECK_GT(rtcfg.data_num_blocks, 0);
  CHECK_GT(rtcfg.data_num_threads, 0);
  const Idx N = csr.row_offsets.length - 1;
  const int ty = 1;
  const int ny = ty * PER_THREAD_WORKLOAD;
  const int by = std::min((N + ny - 1) / ny, static_cast<Idx>(MAX_NBLOCKS));
  const dim3 nblks(rtcfg.data_num_blocks, by);
  const dim3 nthrs(rtcfg.data_num_threads, ty);
  /*
  LOG(INFO) << "Blocks: (" << nblks.x << "," << nblks.y << ") Threads: ("
    << nthrs.x << "," << nthrs.y << ")";
  */
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
    AdvanceAlg algo,
    const RuntimeConfig& rtcfg,
    const SpMat<Idx> &spmat,
    GData* gdata,
    IntArray1D<Idx>* output_frontier,
    Alloc* alloc) {
  /*
  Idx out_len = coo.column.length;
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
   */
  IntArray1D<Idx> outbuf = IntArray1D<Idx>(); //(output_frontier)? *output_frontier : IntArray1D<Idx>();
  switch (algo) {
    case kGunrockLBOut :
      switch (Config::kParallel) {
        case kSrc:
          CudaAdvanceAllNodeParallel<Idx, DType, Config, GData, Functor, Alloc>(
              rtcfg, *spmat.csr, gdata, outbuf, alloc);
          break;
        case kEdge:
          CudaAdvanceAllGunrockLBOut<Idx, DType, Config, GData, Functor, Alloc>(
              rtcfg, *spmat.coo, gdata, outbuf, alloc);
          break;
        case kDst:
          CudaAdvanceAllNodeParallel<Idx, DType, Config, GData, Functor, Alloc>(
              rtcfg, *spmat.csr_t, gdata, outbuf, alloc);
          break;
      }
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
