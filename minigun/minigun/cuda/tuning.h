#ifndef MINIGUN_CUDA_TUNING_H_
#define MINIGUN_CUDA_TUNING_H_

#include "../advance.h"

namespace minigun {
namespace advance {

template <typename Idx, typename Config>
AdvanceAlg FindAdvanceAllAlgo(
    const RuntimeConfig& rtcfg,
    const Coo<Idx>& coo) {
  // TODO(minjie): more
  return kGunrockLBOut;
}

template <typename Idx, typename Config>
AdvanceAlg FindAdvanceAlgo(
    const RuntimeConfig& rtcfg,
    const Coo<Idx>& coo,
    const IntArray1D<Idx>& input_frontier) {
  // TODO(minjie): more
  return kGunrockLBOut;
}

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_TUNING_H_
