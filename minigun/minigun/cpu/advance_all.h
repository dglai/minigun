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
  const Idx E = coo.column.length;
#pragma omp parallel for
  for (Idx eid = 0; eid < E; ++eid) {
    const Idx src = coo.row.data[eid];
    const Idx dst = coo.column.data[eid];
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
  const Idx N = csr.row_offsets.length - 1;
  const Idx feat_size = Functor::GetFeatSize(gdata);
  DType *outbuf = Functor::GetOutBuf(gdata);
#pragma omp parallel
  {
    DType val = static_cast<DType>(0);
    if (Config::kParallel == kDst) {
#pragma omp for
      for (Idx vid = 0; vid < N; ++vid) {
        const Idx dst = vid;
        const Idx start = csr.row_offsets.data[dst];
        const Idx end = csr.row_offsets.data[dst + 1];
        if (start < end) {
          for (Idx feat_idx = 0; feat_idx < feat_size; ++feat_idx) {
            const Idx outoff = Functor::GetOutOffset(dst, gdata) * feat_size + feat_idx;
            if (outbuf != nullptr)
              val = outbuf[outoff];
            for (Idx eid = start; eid < end; ++eid) {
              const Idx src = csr.column_indices.data[eid];
              if (Functor::CondEdge(src, dst, eid, gdata)) {
                Functor::ApplyEdgeReduce(src, dst, eid, feat_idx, val, gdata);
              }
            }
            if (outbuf != nullptr)
              outbuf[outoff] = val;
          }
        }
      }
    } else {
#pragma omp for
      for (Idx vid = 0; vid < N; ++vid) {
        const Idx src = vid;
        const Idx start = csr.row_offsets.data[src];
        const Idx end = csr.row_offsets.data[src + 1];
        if (start < end) {
          for (Idx feat_idx = 0; feat_idx < feat_size; ++feat_idx) {
            const Idx outoff = Functor::GetOutOffset(src, gdata) * feat_size + feat_idx;
            if (outbuf != nullptr)
              val = outbuf[outoff];
            for (Idx eid = start; eid < end; ++eid) {
              const Idx dst = csr.column_indices.data[eid];
              if (Functor::CondEdge(src, dst, eid, gdata)) {
                Functor::ApplyEdgeReduce(src, dst, eid, feat_idx, val, gdata);
              }
            }
            if (outbuf != nullptr)
              outbuf[outoff] = val;
          }
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
      CPUAdvanceAllNodeParallel<Idx, DType, Config, GData, Functor, Alloc>
          (*spmat.csr, gdata);
      break;
    case kEdge:
      CPUAdvanceAllEdgeParallel<Idx, DType, Config, GData, Functor, Alloc>
          (*spmat.coo, gdata);
      break;
    case kDst:
      CPUAdvanceAllNodeParallel<Idx, DType, Config, GData, Functor, Alloc>
          (*spmat.csr_t, gdata);
      break;
  }
}


} //namespace advance
} //namespace minigun

#endif //MINIGUN_ADVANCE_ALL_H
