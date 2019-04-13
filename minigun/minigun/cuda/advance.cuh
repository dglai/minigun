#ifndef MINIGUN_CUDA_ADVANCE_CUH_
#define MINIGUN_CUDA_ADVANCE_CUH_

#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>

#include "../advance.h"
#include "./advance_all.cuh"
#include "./cuda_common.h"
#include "./tuning.h"

namespace minigun {
namespace advance {

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
      if (Config::kMode != kV2N && Config::kMode != kE2N
          && output_frontier.data == nullptr) {
        edge_counts.length = input_frontier.length;
        edge_counts.data = alloc.template AllocateWorkspace<mg_int>(
            edge_counts.length * sizeof(mg_int));
        ComputeEdgeCounts(mgpuctx, csr, input_frontier, &edge_counts);
        lcl_row_offsets.length = input_frontier.length + 1;
        lcl_row_offsets.data = alloc.template AllocateWorkspace<mg_int>(
            lcl_row_offsets.length * sizeof(mg_int));
        ComputeOutputLength(mgpuctx,
            csr, edge_counts, &lcl_row_offsets, &output_frontier.length);
        // The output frontier buffer should be allocated.
        output_frontier.data = alloc.template AllocateData<mg_int>(
            output_frontier.length * sizeof(mg_int));
      }
    }

    // Call advance
    if (Config::kAdvanceAll) {
      AdvanceAlg algo = FindAdvanceAllAlgo<Config>(rtcfg, csr);
      CudaAdvanceAll<Config, GData, Functor, Alloc>(
          algo, rtcfg, csr, gdata, output_frontier);
    } else {
      AdvanceAlg algo = FindAdvanceAlgo<Config>(rtcfg, csr,
          input_frontier, output_frontier);
      LOG(FATAL) << "!!!";
    }

    if (edge_counts.data) {
      alloc.FreeWorkspace(edge_counts.data);
    }
    if (lcl_row_offsets.data) {
      alloc.FreeWorkspace(lcl_row_offsets.data);
    }
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_CUH_
