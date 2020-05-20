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

struct RuntimeConfig {
  // device context
  DLContext ctx;
  // number of thread blocks to process data dimension
  int data_num_blocks = 0;
  // number of threads per block to process data dimension
  int data_num_threads = 0;
#ifdef __CUDACC__
  cudaStream_t stream{nullptr};
#endif  // __CUDACC__
};

enum ParallelMode {
  kSrc = 0, //  Node parallel(by source).
  kEdge,    //  Edge parallel.
  kDst,     //  Node parallel(by destination).
};

// Static config of advance kernel
template <ParallelMode PARALLEL>
struct Config {
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
      const SpMat<Idx>& spmat,
      GData* gdata,
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
             Alloc* alloc = DefaultAllocator<XPU>::Get()) {
  DispatchXPU<XPU, Idx, DType, Config, GData, Functor, Alloc>::Advance(
      config, spmat, gdata, alloc);
}

}  // namespace advance
}  // namespace minigun

#ifdef __CUDACC__
#include "./cuda/advance.cuh"
#else
#include "./cpu/advance.h"
#endif

#endif  // MINIGUN_ADVANCE_H_
