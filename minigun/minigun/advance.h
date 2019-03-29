#ifndef MINIGUN_ADVANCE_H_
#define MINIGUN_ADVANCE_H_

#include "./base.h"
#include "./csr.h"
#ifdef MINIGUN_USE_CUDA
#include <cuda_runtime.h>
#endif  // MINIGUN_USE_CUDA

namespace minigun {
namespace advance {

struct RuntimeConfig {
#ifdef MINIGUN_USE_CUDA
  cudaStream_t stream{nullptr};
#endif  // MINIGUN_USE_CUDA
};

struct DefaultAllocator {
  // TODO
};

template <int XPU,
          typename VFrame,
          typename EFrame,
          typename Functor,
          typename Alloc>
struct DispatchXPU {
  static void Advance(
      const RuntimeConfig& config,
      const Csr& csr,
      VFrame vframe,
      EFrame eframe,
      IntArray1D input_frontier,
      IntArray1D output_frontier,
      const Alloc& alloc) {
    LOG(FATAL) << "Not implemented for XPU: " << XPU;
  }
};

template <typename VFrame,
          typename EFrame,
          typename Functor,
          typename Alloc = DefaultAllocator>
void Advance(const RuntimeConfig& config,
             const Csr& csr,
             VFrame vframe,
             EFrame eframe,
             IntArray1D input_frontier,
             IntArray1D output_frontier,
             const Alloc& alloc = Alloc()) {
  DispatchXPU<kDLGPU, VFrame, EFrame, Functor, Alloc>::Advance(
      config, csr, vframe, eframe,
      input_frontier, output_frontier, alloc);
}

}  // namespace advance
}  // namespace minigun

#ifdef MINIGUN_USE_CUDA
#include "./cuda/advance.cuh"
#endif

#endif  // MINIGUN_ADVANCE_H_
