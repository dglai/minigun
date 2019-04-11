#ifndef MINIGUN_ADVANCE_H_
#define MINIGUN_ADVANCE_H_

#include "./base.h"
#include "./csr.h"
#ifdef MINIGUN_USE_CUDA
#include <cuda_runtime.h>
#endif  // MINIGUN_USE_CUDA

#define AT_CASE_DEV_TYPE(dev_type)                                      \
  case dev_type: {                                                      \
    DispatchXPU<dev_type, GData, Functor, Alloc>::Advance(              \
        config, csr, gdata, input_frontier, output_frontier, alloc);    \
    break;                                                              \
  }

namespace minigun {
namespace advance {

enum AdvanceAlg {
  kAuto = 0,  // auto-tuning
  kAllEdges,
  kLoadBalance,
  kTWC,
};

struct RuntimeConfig {
  AdvanceAlg alg = kAuto;
  // number of thread blocks to process data dimension
  int data_num_blocks = 0;
  // number of threads per block to process data dimension
  int data_num_threads = 0;
#ifdef MINIGUN_USE_CUDA
  cudaStream_t stream{nullptr};
#endif  // MINIGUN_USE_CUDA
};

struct DefaultAllocator {
  // TODO
};

template <int XPU,
          typename GData,
          typename Functor,
          typename Alloc>
struct DispatchXPU {
  static void Advance(
      const RuntimeConfig& config,
      const Csr& csr,
      GData* gdata,
      IntArray1D input_frontier,
      IntArray1D output_frontier,
      const Alloc& alloc) {
    LOG(FATAL) << "Not implemented for XPU: " << XPU;
  }
};

template <typename GData,
          typename Functor,
          typename Alloc = DefaultAllocator>
void Advance(const RuntimeConfig& config,
             const Csr& csr,
             GData* gdata,
             IntArray1D input_frontier,
             IntArray1D output_frontier,
             const Alloc& alloc = Alloc()) {
  switch (csr.ctx.device_type) {
    AT_CASE_DEV_TYPE(kDLCPU)
    default:
      LOG(FATAL) << "Device type not supported: " << csr.ctx.device_type;
  }
}

}  // namespace advance
}  // namespace minigun

#ifdef MINIGUN_USE_CUDA
#include "./cuda/advance.cuh"
#endif
#include "./cpu/advance.h"

#endif  // MINIGUN_ADVANCE_H_
