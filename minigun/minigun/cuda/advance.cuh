#ifndef MINIGUN_CUDA_ADVANCE_CUH_
#define MINIGUN_CUDA_ADVANCE_CUH_

#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>

#include "../advance.h"
#include "./advance_all.cuh"
#include "./advance_lb.cuh"
#include "./cuda_common.h"
#include "./tuning.h"

namespace minigun {
namespace advance {


template <typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
class CudaAdvanceExecutor {
 public:
  CudaAdvanceExecutor(
      AdvanceAlg algo,
      const RuntimeConfig& rtcfg,
      const Csr& csr,
      GData* gdata,
      IntArray1D input_frontier,
      IntArray1D output_frontier,
      Alloc alloc):
    algo_(algo), rtcfg_(rtcfg), csr_(csr), gdata_(gdata),
    input_frontier_(input_frontier), output_frontier_(output_frontier),
    alloc_(alloc) { }

  void Run() {
    MgpuContext mgpuctx(rtcfg_.stream);
    // Row offset array of local graph (graph sliced by nodes in input_frontier).
    // It's length == len(input_frontier) + 1, and
    // lcl_row_offsets[i+1] - lcl_row_offsets[i] == edge_counts[i]
    IntArray1D lcl_row_offsets;
  
    if (Config::kMode != kV2N && Config::kMode != kE2N
        && output_frontier_.data == nullptr) {
      lcl_row_offsets.length = input_frontier_.length + 1;
      lcl_row_offsets.data = alloc_.template AllocateWorkspace<mg_int>(
          lcl_row_offsets.length * sizeof(mg_int));
      ComputeOutputLength(mgpuctx, &lcl_row_offsets, &output_frontier_.length);
      // The output frontier buffer should be allocated.
      output_frontier_.data = alloc_.template AllocateData<mg_int>(
          output_frontier_.length * sizeof(mg_int));
    }

    switch (algo_) {
      case kGunrockLBOut :
        CudaAdvanceGunrockLBOut(mgpuctx, lcl_row_offsets);
        break;
      default:
        LOG(FATAL) << "Algorithm " << algo_ << " is not supported.";
    }

    if (lcl_row_offsets.data) {
      alloc_.FreeWorkspace(lcl_row_offsets.data);
    }
  }

  void ComputeOutputLength(MgpuContext& mgpuctx,
      IntArray1D* lcl_row_offsets, mg_int* outlen) {
    // An array of size len(input_frontier). Each element i is the number of
    // edges input_frointer[i] has.
    IntArray1D edge_counts;
    edge_counts.length = input_frontier_.length;
    edge_counts.data = alloc_.template AllocateWorkspace<mg_int>(
        edge_counts.length * sizeof(mg_int));
    mgpu::transform(
      [] MGPU_DEVICE(int tid, const mg_int* rowoff, const mg_int* infront, mg_int* ecounts) {
        const mg_int vid = infront[tid];
        ecounts[tid] = rowoff[vid + 1] - rowoff[vid];
      }, edge_counts.length, mgpuctx,
      csr_.row_offsets.data,
      input_frontier_.data,
      edge_counts.data);
    // compute output len using prefix scan
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
    alloc_.FreeWorkspace(edge_counts.data);
  }

  void CudaAdvanceGunrockLBOut(MgpuContext& mgpuctx, IntArray1D lcl_row_offsets) {
  }

 private:
  const AdvanceAlg algo_;
  const RuntimeConfig& rtcfg_;
  const Csr& csr_;
  GData* gdata_;
  IntArray1D input_frontier_;
  IntArray1D output_frontier_;
  Alloc alloc_;
};

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
    // Call advance
    if (Config::kAdvanceAll) {
      AdvanceAlg algo = FindAdvanceAllAlgo<Config>(rtcfg, csr);
      CudaAdvanceAll<Config, GData, Functor, Alloc>(
          algo, rtcfg, csr, gdata, output_frontier, alloc);
    } else {
      AdvanceAlg algo = FindAdvanceAlgo<Config>(rtcfg, csr,
          input_frontier, output_frontier);
      CudaAdvanceExecutor<Config, GData, Functor, Alloc> exec(
          algo, rtcfg, csr, gdata, input_frontier, output_frontier, alloc);
      exec.Run();
    }
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_CUH_
