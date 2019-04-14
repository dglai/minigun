#ifndef MINIGUN_ADVANCE_H_
#define MINIGUN_ADVANCE_H_

#include "./base.h"
#include "./csr.h"
#include "./mem.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif  // __CUDACC__

namespace minigun {
namespace advance {

enum AdvanceAlg {
  kAuto = 0,  // auto-tuning
  // Gunrock's LB policy: parallelize by output frontiers
  kGunrockLBOut,
  kTWC,
};

struct RuntimeConfig {
  // device context
  DLContext ctx;
  // the advance algorithm to use
  AdvanceAlg alg = kAuto;
  // number of thread blocks to process data dimension
  int data_num_blocks = 0;
  // number of threads per block to process data dimension
  int data_num_threads = 0;
#ifdef __CUDACC__
  cudaStream_t stream{nullptr};
#endif  // __CUDACC__
};

// Different frontier mode
enum FrontierMode {
  kV2N = 0,  // in front contains vids, no out front
  kV2E,      // in front contains vids, out front contains eids
  kV2V,      // in front contains vids, out front contains vids
  kE2N,      // in front contains eids, no out front
  kE2E,      // in front contains eids, out front contains eids
  kE2V,      // in front contains eids, out front contains vids
};

// Static config of advance kernel
template <bool ADVANCE_ALL,
          FrontierMode MODE>
struct Config {
  // if true, the advance is applied on all the nodes
  static const bool kAdvanceAll = ADVANCE_ALL;
  // frontier mode
  static const FrontierMode kMode = MODE;
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
      Alloc* alloc) {
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
             Alloc* alloc = DefaultAllocator<XPU>::Get()) {
  DispatchXPU<XPU, Config, GData, Functor, Alloc>::Advance(
      config, csr, gdata,
      input_frontier, output_frontier, alloc);
}

}  // namespace advance
}  // namespace minigun

#ifdef __CUDACC__
#include "./cuda/advance.cuh"
#else
#include "./cpu/advance.h"
#endif

#endif  // MINIGUN_ADVANCE_H_
