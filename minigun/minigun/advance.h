#ifndef MINIGUN_ADVANCE_H_
#define MINIGUN_ADVANCE_H_

#include "./base.h"
#include "./csr.h"
#include "./mem.h"
#ifdef MINIGUN_USE_CUDA
#include <cuda_runtime.h>
#endif  // MINIGUN_USE_CUDA

namespace minigun {
namespace advance {

enum AdvanceAlg {
  kAuto = 0,  // auto-tuning
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

template <bool ADVANCE_ALL,
          bool REQUIRE_OUT_FRONT>
struct Config {
  // if true, the advance is applied on all the nodes
  static const bool kAdvanceAll = ADVANCE_ALL;
  // if true, the kernel needs to generate output froniter
  static const bool kRequireOutFront = REQUIRE_OUT_FRONT;
};

template <int XPU,
          typename Config,
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
      Alloc alloc) {
    LOG(FATAL) << "Not implemented for XPU: " << XPU;
  }
};


/*
 * !\brief Advance kernel.
 */
template <int XPU,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc = DefaultAllocator<XPU> >
void Advance(const RuntimeConfig& config,
             const Csr& csr,
             GData* gdata,
             IntArray1D input_frontier,
             IntArray1D output_frontier,
             Alloc alloc = Alloc()) {
  DispatchXPU<XPU, Config, GData, Functor, Alloc>::Advance(
      config, csr, gdata,
      input_frontier, output_frontier, alloc);
}

}  // namespace advance
}  // namespace minigun

#ifdef MINIGUN_USE_CUDA
#include "./cuda/advance.cuh"
#endif

#endif  // MINIGUN_ADVANCE_H_
