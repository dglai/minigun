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
  mg_int ty = blockIdx.y * blockDim.y + threadIdx.y;
  mg_int stride_y = blockDim.y * gridDim.y;
  mg_int eid = ty;
  while (eid < csr.column_indices.length) {
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

__inline__ void ComputeEdgeCounts(
    MgpuContext& mgpuctx,
    const Csr& csr,
    const IntArray1D& input_frontier,
    IntArray1D* edge_counts) {
  mgpu::transform(
    [] MGPU_DEVICE(int tid, const mg_int* rowoff, const mg_int* infront, mg_int* ecounts) {
      const mg_int vid = infront[tid];
      ecounts[tid] = rowoff[vid + 1] - rowoff[vid];
    }, edge_counts->length, mgpuctx,
    csr.row_offsets.data,
    input_frontier.data,
    edge_counts->data);
}

__inline__ void ComputeOutputLength(
    MgpuContext& mgpuctx,
    const Csr& csr,
    const IntArray1D& edge_counts,
    IntArray1D* lcl_row_offsets,
    mg_int* outlen) {
  mgpu::scan<mgpu::scan_type_exc>(
      edge_counts.data,
      edge_counts.length,
      lcl_row_offsets->data,
      mgpu::plus_t<mg_int>(),
      lcl_row_offsets->data + edge_counts.length,  // the last element stores the reduction.
      mgpuctx);
  // get the reduction
  CUDA_CALL(cudaMemcpy(outlen, lcl_row_offsets->data + edge_counts.length,
        sizeof(mg_int), cudaMemcpyDeviceToHost));
  LOG(INFO) << "Output frontier length: " << *outlen;
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
    // An array of size len(input_frontier). Each element i is the number of
    // edges input_frointer[i] has.
    IntArray1D edge_counts;
    // Row offset array of local graph (graph sliced by nodes in input_frontier).
    // It's length == len(input_frontier) + 1, and
    // lcl_row_offsets[i+1] - lcl_row_offsets[i] == edge_counts[i]
    IntArray1D lcl_row_offsets;

    if (Config::kAdvanceAll) {
      // input_frontier and edge_counts will not be used.
      if (Config::kMode != kV2N && Config::kMode != kE2N
          && output_frontier.data == nullptr) {
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
      ComputeEdgeCounts(mgpuctx, csr, input_frontier, &edge_counts);
      lcl_row_offsets.length = input_frontier.length + 1;
      lcl_row_offsets.data = alloc.template AllocateWorkspace<mg_int>(
          lcl_row_offsets.length * sizeof(mg_int));
      ComputeOutputLength(mgpuctx,
          csr, edge_counts, &lcl_row_offsets, &output_frontier.length);
      if (Config::kMode != kV2N && Config::kMode != kE2N
          && output_frontier.data == nullptr) {
        // The output frontier buffer should be allocated.
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
