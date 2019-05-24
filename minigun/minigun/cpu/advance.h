#ifndef MINIGUN_CPU_ADVANCE_H_
#define MINIGUN_CPU_ADVANCE_H_

#include "../advance.h"
#include <dmlc/omp.h>

namespace minigun {
namespace advance {

template <typename Idx, typename Alloc>
Idx ComputeOutputLength(Csr<Idx> csr,
                        IntArray1D<Idx> input_frontier,
                        IntArray1D<Idx>* lcl_row_offsets,
                        Alloc* alloc) {
  IntArray1D<Idx> edge_counts;
  lcl_row_offsets->data[0] = 0;
  edge_counts.length = input_frontier.length;
  edge_counts.data = alloc->template AllocateWorkspace<Idx>(
      edge_counts.length * sizeof(Idx));
#pragma omp parallel for
  for (Idx tid = 0; tid < edge_counts.length; ++tid) {
    const Idx vid = input_frontier.data[tid];
    edge_counts.data[tid] = csr.row_offsets.data[vid + 1] -
      csr.row_offsets.data[vid];
  }

  // prefix sum with openmp
  Idx* thread_sum;
#pragma omp parallel
  {
    const Idx tid = omp_get_thread_num();
    const Idx ntr = omp_get_num_threads();

    // only one thread should allocate workspace
#pragma omp single
    {
      thread_sum = alloc->template AllocateWorkspace<Idx>(ntr *
          sizeof(Idx));
    }

    // each thread calculates one chunk of partial sum
    Idx sum = 0;
#pragma omp for schedule(static)
    for (Idx vid = 0; vid < edge_counts.length; ++vid) {
      sum += edge_counts.data[vid];
      lcl_row_offsets->data[vid + 1] = sum;
    }
    thread_sum[tid] = sum;
#pragma omp barrier
    // fix partial sum in each chunk by adding offsets
    Idx offset = 0;
    for (Idx i = 0; i < tid; ++i) {
      offset += thread_sum[i];
    }
#pragma omp for schedule(static)
    for (Idx vid = 0; vid < edge_counts.length; ++vid) {
      lcl_row_offsets->data[vid + 1] += offset;
    }
  }

  alloc->FreeWorkspace(edge_counts.data);
  alloc->FreeWorkspace(thread_sum);
  return lcl_row_offsets->data[edge_counts.length];
}

template <typename Idx,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CPUAdvance(Csr<Idx> csr,
                GData* gdata,
                IntArray1D<Idx> input_frontier,
                IntArray1D<Idx> output_frontier,
                IntArray1D<Idx> lcl_row_offsets,
                Alloc* alloc) {
  Idx N = Config::kAdvanceAll ? csr.row_offsets.length - 1 : input_frontier.length;
#pragma omp parallel for
  for (Idx vid = 0; vid < N; ++vid) {
    Idx src = vid;
    if (!Config::kAdvanceAll) {
      src = input_frontier.data[vid];
    }
    const Idx row_start = csr.row_offsets.data[src];
    const Idx row_end = csr.row_offsets.data[src + 1];
    for (Idx eid = row_start; eid < row_end; ++eid) {
      const Idx dst = csr.column_indices.data[eid];
      if (Functor::CondEdge(src, dst, eid, gdata)) {
        Functor::ApplyEdge(src, dst, eid, gdata);
        if (Config::kMode != kV2N && Config::kMode != kE2N) {

          Idx out_idx;
          if (Config::kAdvanceAll) {
            out_idx = eid;
          } else {
            out_idx = eid - row_start + lcl_row_offsets.data[vid];
          }
          if (Config::kMode == kV2V || Config::kMode == kE2V) {
            output_frontier.data[out_idx] = dst;
          } else {
            output_frontier.data[out_idx] = eid;
          }
        }
      } else {
        if (Config::kMode != kV2N && Config::kMode != kE2N) {
          Idx out_idx;
          if (Config::kAdvanceAll) {
            out_idx = eid;
          } else {
            out_idx = eid - row_start + lcl_row_offsets.data[vid];
          }
          output_frontier.data[out_idx] = MG_INVALID;
        }
      }
    }
  }
}

template <typename Idx,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
struct DispatchXPU<kDLCPU, Idx, Config, GData, Functor, Alloc> {
  static void Advance(
      const RuntimeConfig& rtcfg,
      const Csr<Idx>& csr,
      GData* gdata,
      IntArray1D<Idx> input_frontier,
      IntArray1D<Idx>* output_frontier,
      Alloc* alloc) {
    if (Config::kMode != kV2V && Config::kMode != kV2E
        && Config::kMode != kV2N) {
      LOG(FATAL) << "Advance from edge not supported for CPU";
    }
    IntArray1D<Idx> lcl_row_offsets;
    Idx out_len = 0;
    if (Config::kAdvanceAll) {
      lcl_row_offsets = csr.row_offsets;
      out_len = csr.column_indices.length;
    } else {
      if (Config::kMode != kV2N && Config::kMode != kE2N) {
        lcl_row_offsets.length = input_frontier.length + 1;
        lcl_row_offsets.data = alloc->template AllocateWorkspace<Idx>(
            lcl_row_offsets.length * sizeof(Idx));
        out_len = ComputeOutputLength(
            csr, input_frontier, &lcl_row_offsets, alloc);
      }
    }
    if (output_frontier) {
      if (output_frontier->data == nullptr) {
        // The output frontier buffer should be allocated.
        output_frontier->length = out_len;
        output_frontier->data = alloc->template AllocateData<Idx>(
            output_frontier->length * sizeof(Idx));
      } else {
        CHECK_GE(output_frontier->length, out_len)
          << "Require output frontier of length " << out_len
          << " but only got a buffer of length " << output_frontier->length;
      }
    }

    IntArray1D<Idx> outbuf = (output_frontier)? *output_frontier : IntArray1D<Idx>();
    CPUAdvance<Idx, Config, GData, Functor, Alloc>(
        csr, gdata, input_frontier, outbuf, lcl_row_offsets, alloc);

    if (!Config::kAdvanceAll && Config::kMode != kV2N
        && Config::kMode != kE2N) {
      alloc->FreeWorkspace(lcl_row_offsets.data);
    }
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CPU_ADVANCE_H_
