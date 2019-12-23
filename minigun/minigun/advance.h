#ifndef MINIGUN_ADVANCE_H_
#define MINIGUN_ADVANCE_H_

#include "./base.h"
#include "spmat.h"
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

enum ParallelMode {
  kSrc = 0, // Node parallel(by source).
  kEdge,    // Edge parallel.
  kDst,     // Node parallel(by destination).
};

// Static config of advance kernel
template <bool ADVANCE_ALL,
          FrontierMode MODE,
          ParallelMode PARALLEL>
struct Config {
  // if true, the advance is applied on all the nodes
  static const bool kAdvanceAll = ADVANCE_ALL;
  // frontier mode
  static const FrontierMode kMode = MODE;
  static const ParallelMode kParallel = PARALLEL;
};

/*!
 * \brief Template specialization to dispatch to implementations
 *        on different devices.
 */
template <int XPU,
          typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
struct DispatchXPU {
  static void Advance(
      const RuntimeConfig& config,
      const Csr<Idx>& csr,
      const Csr<Idx>& csr_t,
      const Coo<Idx>& coo,
      GData* gdata,
      IntArray1D<Idx> input_frontier,
      IntArray1D<Idx>* output_frontier,
      Alloc* alloc) {
    LOG(FATAL) << "Not implemented for XPU: " << XPU;
  }
};


/*!
 * \brief Advance kernel.
 *
 * \tparam XPU The computing device type (DLDeviceType)
 * \tparam Idx The type of the index (usually int32_t or int64_t)
 * \tparam Config The static configuration of advance kernel.
 * \tparam GData The user-defined graph data.
 * \tparam Functor The user-defined functions.
 * \tparam Alloc The external allocator type.
 * \param config Runtime configuration of this advance kernel.
 * \param csr The graph csr structure.
 * \param gdata The pointer to the user-defined graph data structure.
 *              This pointer must be a host pointer and it will be
 *              dereferenced and its content will be copied to the
 *              device for execution.
 * \param input_frontier The input frontier array. Could be empty.
 * \param output_frontier The pointer to the output frontier array.
 * \param alloc The external memory allocator.
 */
template <int XPU,
          typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc = DefaultAllocator<XPU> >
void Advance(const RuntimeConfig& config,
             const SpMat<Idx>& spmat,
             GData* gdata,
             IntArray1D<Idx> input_frontier,
             IntArray1D<Idx>* output_frontier = nullptr,
             Alloc* alloc = DefaultAllocator<XPU>::Get()) {
  if (Config::kMode != kV2N && Config::kMode != kE2N
      && output_frontier == nullptr) {
    LOG(FATAL) << "Require computing output frontier but no buffer is provided.";
  }
  DispatchXPU<XPU, Idx, DType, Config, GData, Functor, Alloc>::Advance(
      config, spmat, gdata,
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
