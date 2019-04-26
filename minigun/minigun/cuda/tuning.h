#ifndef MINIGUN_CUDA_TUNING_H_
#define MINIGUN_CUDA_TUNING_H_

#include "../advance.h"

namespace minigun {
namespace advance {

template <typename Config>
AdvanceAlg FindAdvanceAllAlgo(
    const RuntimeConfig& rtcfg,
    const Csr& csr) {
  // TODO(minjie): more
  return kGunrockLBOut;
}

template <typename Config>
AdvanceAlg FindAdvanceAlgo(
    const RuntimeConfig& rtcfg,
    const Csr& csr,
    const IntArray1D& input_frontier) {
  // TODO(minjie): more
  return kGunrockLBOut;
}

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_TUNING_H_
