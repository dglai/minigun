#ifndef MINIGUN_CUDA_ADVANCE_CUH_
#define MINIGUN_CUDA_ADVANCE_CUH_

#include "../advance.h"
#include "./advance_all.cuh"
#include "./cuda_common.cuh"
#include "./tuning.h"


namespace minigun {
namespace advance {

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
struct DispatchXPU<kDLGPU, Idx, DType, Config, GData, Functor, Alloc> {
  static void Advance(
      const RuntimeConfig& rtcfg,
      const SpMat<Idx>& spmat,
      GData* gdata,
      IntArray1D<Idx> input_frontier,
      IntArray1D<Idx>* output_frontier,
      Alloc* alloc) {
    // Call advance
    if (Config::kAdvanceAll) {
      AdvanceAlg algo = FindAdvanceAllAlgo<Idx, Config>(rtcfg, spmat);
      CudaAdvanceAll<Idx, DType, Config, GData, Functor, Alloc>(
          algo, rtcfg, spmat, gdata, alloc);
    } else {
      LOG(FATAL) << "Partial frontier is not supported yet.";
    }
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_CUH_
