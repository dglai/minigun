#ifndef MINIGUN_CUDA_ADVANCE_CUH_
#define MINIGUN_CUDA_ADVANCE_CUH_

#include "../advance.h"
#include "./advance_all.cuh"
#include "./cuda_common.cuh"

#if ENABLE_PARTIAL_FRONTIER
#include "./advance_lb.cuh"
#endif

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
      Alloc* alloc) {
    // Call advance
    CudaAdvanceAll<Idx, DType, Config, GData, Functor, Alloc>(
        rtcfg, spmat, gdata, alloc);
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_CUH_