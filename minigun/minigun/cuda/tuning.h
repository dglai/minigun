#ifndef MINIGUN_CUDA_TUNING_H_
#define MINIGUN_CUDA_TUNING_H_

#include "../advance.h"

namespace minigun {
namespace advance {

#define MAX_BLOCK_NTHREADS 1024
#define PER_THREAD_WORKLOAD 1
#define MAX_NBLOCKS 65535L

struct KernelConfig {
  AdvanceAlg alg;
  int by, ty;
};

template <typename Config>
void TuneKernelConfig(
    const RuntimeConfig& rtcfg,
    const Csr& csr,
    const IntArray1D& input_frontier,
    const IntArray1D& output_frontier,
    KernelConfig* kcfg) {
  // first, find which algorithm to use
  if (rtcfg.alg != kAuto) {
    kcfg->alg = rtcfg.alg;
  }
  kcfg->alg = kGunrockLBOut; // XXX: only one kernel right now.

  // second, find the kernel config for each algorithm
  if (kcfg->alg == kGunrockLBOut) {
    const mg_int M = Config::kAdvanceAll?
      csr.column_indices.length :
      output_frontier.length;
    kcfg->ty = MAX_BLOCK_NTHREADS / rtcfg.data_num_threads;
    const int ny = kcfg->ty * PER_THREAD_WORKLOAD;
    kcfg->by = std::min((M + ny - 1) / ny, MAX_NBLOCKS);
  } else {
    LOG(FATAL) << "Unknown advance kernel algorithm: " << kcfg->alg;
  }
}

#undef MAX_BLOCK_NTHREADS
#undef PER_THREAD_WORKLOAD
#undef MAX_NBLOCKS

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_TUNING_H_
