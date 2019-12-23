#ifndef MINIGUN_CPU_ADVANCE_H_
#define MINIGUN_CPU_ADVANCE_H_

#include "../advance.h"
#include "./advance_all.h"
#include <dmlc/omp.h>

namespace minigun {
namespace advance {

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
struct DispatchXPU<kDLCPU, Idx, DType, Config, GData, Functor, Alloc> {
  static void Advance(
      const RuntimeConfig& rtcfg,
      const SpMat<Idx>& spmat,
      GData* gdata,
      IntArray1D<Idx> input_frontier,
      IntArray1D<Idx>* output_frontier,
      Alloc* alloc) {
    if (Config::kAdvanceAll) {
      CPUAdvanceAll<Idx, DType, Config, GData, Functor, Alloc>(
          spmat, gdata);
    } else {
      LOG(FATAL) << "Partial frontier is not supported yet.";
    }
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CPU_ADVANCE_H_
