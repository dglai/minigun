//
// Created by Ye, Zihao on 2019-11-23.
//

#ifndef MINIGUN_ADVANCE_ALL_H
#define MINIGUN_ADVANCE_ALL_H

#include "../advance.h"
#include <dmlc/omp.h>

namespace minigun {
namespace advance {

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CPUAdvanceAllEdgeParallel(
    const Coo<Idx>& coo,
    GData *gdata) {
  Idx E = coo.column.length;
#pragma omp parallel for
  for (Idx eid = 0; eid < E; ++eid) {
    const Idx src = coo.row.data[eid];
    const Idx dst = coo.col.data[eid];
    if (Functor::CondEdge(src, dst, eid, gdata))
      Functor::ApplyEdge(src, dst, eid, gdata);
  }
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CPUAdvanceAllNodeParallel(
    const Csr<Idx>& csr,
    GData *gdata) {
  Idx N = csr.row_offsets.length - 1;
  if (Config::kParallel == kDst) {
#pragma omp parallel for
    for (Idx vid = 0; vid < N; ++vid) {
      const Idx dst = vid;
      const Idx start = csr.row_offsets.data[dst];
      const Idx end = csr.row_offsets.data[dst + 1];
      for (Idx eid = start; eid < end; ++eid) {
        const Idx src = csr.column_indices.data[eid];
        if (Functor::CondEdge(src, dst, eid, gdata)) {
          Functor::ApplyEdge(src, dst, eid, gdata);
        }
      }
    }
  } else {
#pragma omp parallel for
    for (Idx vid = 0; vid < N; ++vid) {
      const Idx src = vid;
      const Idx start = csr.row_offsets.data[src];
      const Idx end = csr.row_offsets.data[src + 1];
      for (Idx eid = start; eid < end; ++eid) {
        const Idx dst = csr.column_indices.data[eid];
        if (Functor::CondEdge(src, dst, eid, gdata)) {
          Functor::ApplyEdge(src, dst, eid, gdata);
        }
      }
    }
  }
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CPUAdvanceAll(
      const SpMat<Idx>& spmat,
      GData* gdata) {
  switch (Config::kParallel) {
    case kSrc:
      CPUAdvanceAllNodeParallel(*spmat.csr, gdata);
      break;
    case kEdge:
      CPUAdvanceAllEdgeParallel(*spmat.coo, gdata);
      break;
    case kDst:
      CPUAdvanceAllNodeParallel(*spmat.csr_t, gdata);
      break;
  }
}


} //namespace advance
} //namespace minigun

#endif //MINIGUN_ADVANCE_ALL_H
