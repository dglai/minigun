#ifndef MINIGUN_CUDA_ADVANCE_CUH_
#define MINIGUN_CUDA_ADVANCE_CUH_

#include "../advance.h"
#include "./advance_all.cuh"
#include "./cuda_common.cuh"
#include "./tuning.h"

#if ENABLE_PARTIAL_FRONTIER
#include "./advance_lb.cuh"
#endif

namespace minigun {
namespace advance {

template <typename Idx,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
struct DispatchXPU<kDLGPU, Idx, Config, GData, Functor, Alloc> {
  static void Advance(
      const RuntimeConfig& rtcfg,
      const Csr<Idx>& csr,
      const Csr<Idx>& csc,
      const Coo<Idx>& coo,
      GData* gdata,
      IntArray1D<Idx> input_frontier,
      IntArray1D<Idx>* output_frontier,
      Alloc* alloc) {
    // Call advance
    if (Config::kAdvanceAll) {
      AdvanceAlg algo = FindAdvanceAllAlgo<Idx, Config>(rtcfg, coo);
      CudaAdvanceAll<Idx, Config, GData, Functor, Alloc>(
          algo, rtcfg, csr, csc, coo, gdata, output_frontier, alloc);
    } else {
#if ENABLE_PARTIAL_FRONTIER
      AdvanceAlg algo = FindAdvanceAlgo<Idx, Config>(rtcfg, coo,
          input_frontier);
      CudaAdvanceExecutor<Idx, Config, GData, Functor, Alloc> exec(
          algo, rtcfg, coo, gdata, input_frontier, output_frontier, alloc);
      exec.Run();
#else
      LOG(FATAL) << "Partial frontier is not supported for CUDA 10.0";
#endif
    }
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_CUH_
