#ifndef MINIGUN_CPU_ADVANCE_H_
#define MINIGUN_CPU_ADVANCE_H_

#include "../advance.h"
#include <omp.h>

namespace minigun {
namespace advance {

// Binary search the row_offsets to find the source node of the edge id.
static inline mg_int CPUBinarySearchSrc(const IntArray1D& array, mg_int eid) {
  mg_int lo = 0, hi = array.length - 1;
  while (lo < hi) {
    mg_int mid = (lo + hi) >> 1;
    if (array.data[mid] < eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // INVARIANT: lo == hi
  if (array.data[hi] == eid) {
    return hi;
  } else {
    return hi - 1;
  }
}

template <typename GData,
          typename Functor>
void CPUAdvanceKernel(
    Csr csr,  // pass by value to make sure it is copied to device memory
    GData* gdata,
    IntArray1D input_frontier,
    IntArray1D output_frontier) {
#pragma omp parallel for
  for (mg_int eid = 0; eid < csr.column_indices.length; ++eid) {
    mg_int src = CPUBinarySearchSrc(csr.row_offsets, eid);
    mg_int dst = csr.column_indices.data[eid];
    if (Functor::CondEdge(src, dst, eid, gdata)) {
      Functor::ApplyEdge(src, dst, eid, gdata);
    }
  }
};

template <typename GData,
          typename Functor,
          typename Alloc>
struct DispatchXPU<kDLCPU, GData, Functor, Alloc> {
  static void Advance(
      const RuntimeConfig& rtcfg,
      const Csr& csr,
      GData* gdata,
      IntArray1D input_frontier,
      IntArray1D output_frontier,
      const Alloc& alloc) {
    //CHECK(output_frontier.length != 0);
    CPUAdvanceKernel<GData, Functor>(csr, gdata, input_frontier, output_frontier);
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CPU_ADVANCE_H_
