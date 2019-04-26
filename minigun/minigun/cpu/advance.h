#ifndef MINIGUN_CPU_ADVANCE_H_
#define MINIGUN_CPU_ADVANCE_H_

#include "../advance.h"
#include <dmlc/omp.h>

namespace minigun {
namespace advance {

template <typename Alloc>
mg_int ComputeOutputLength(Csr csr,
                           IntArray1D input_frontier,
                           IntArray1D* lcl_row_offsets,
                           Alloc* alloc) {
  IntArray1D edge_counts;
  lcl_row_offsets->data[0] = 0;
  edge_counts.length = input_frontier.length;
  edge_counts.data = alloc->template AllocateWorkspace<mg_int>(
      edge_counts.length * sizeof(mg_int));
#pragma omp parallel for
  for (mg_int tid = 0; tid < edge_counts.length; ++tid) {
    const mg_int vid = input_frontier.data[tid];
    edge_counts.data[tid] = csr.row_offsets.data[vid + 1] -
      csr.row_offsets.data[vid];
  }

  // prefix sum with openmp
  mg_int* thread_sum;
#pragma omp parallel
  {
    const mg_int tid = omp_get_thread_num();
    const mg_int ntr = omp_get_num_threads();

    // only one thread should allocate workspace
#pragma omp single
    {
      thread_sum = alloc->template AllocateWorkspace<mg_int>(ntr *
          sizeof(mg_int));
    }

    // each thread calculates one chunk of partial sum
    mg_int sum = 0;
#pragma omp for schedule(static)
    for (mg_int vid = 0; vid < edge_counts.length; ++vid) {
      sum += edge_counts.data[vid];
      lcl_row_offsets->data[vid + 1] = sum;
    }
    thread_sum[tid] = sum;
#pragma omp barrier
    // fix partial sum in each chunk by adding offsets
    mg_int offset = 0;
    for (mg_int i = 0; i < tid; ++i) {
      offset += thread_sum[i];
    }
#pragma omp for schedule(static)
    for (mg_int vid = 0; vid < edge_counts.length; ++vid) {
      lcl_row_offsets->data[vid + 1] += offset;
    }
  }

  alloc->FreeWorkspace(edge_counts.data);
  alloc->FreeWorkspace(thread_sum);
  return lcl_row_offsets->data[edge_counts.length];
}

template <typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CPUAdvance(Csr csr,
                GData* gdata,
                IntArray1D input_frontier,
                IntArray1D output_frontier,
                IntArray1D lcl_row_offsets,
                Alloc* alloc) {
  mg_int N;
  if (Config::kAdvanceAll) {
    N = csr.row_offsets.length - 1;
  } else {
    N = input_frontier.length;
  }
#pragma omp parallel for
  for (mg_int vid = 0; vid < N; ++vid) {
    mg_int src = vid;
    if (!Config::kAdvanceAll) {
      src = input_frontier.data[vid];
    }
    const mg_int row_start = csr.row_offsets.data[src];
    const mg_int row_end = csr.row_offsets.data[src + 1];
    for (mg_int eid = row_start; eid < row_end; ++eid) {
      const mg_int dst = csr.column_indices.data[eid];
      if (Functor::CondEdge(src, dst, eid, gdata)) {
        Functor::ApplyEdge(src, dst, eid, gdata);
        if (Config::kMode != kV2N && Config::kMode != kE2N) {

          mg_int out_idx;
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
          mg_int out_idx;
          if (Config::kAdvanceAll) {
            out_idx = eid;
          } else {
            out_idx = eid - row_start + lcl_row_offsets.data[vid];
          }
          output_frontier.data[out_idx] = kInvalid;
        }
      }
    }
  }
}

template <typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
struct DispatchXPU<kDLCPU, Config, GData, Functor, Alloc> {
  static void Advance(
      const RuntimeConfig& rtcfg,
      const Csr& csr,
      GData* gdata,
      IntArray1D input_frontier,
      IntArray1D* output_frontier,
      Alloc* alloc) {
    if (Config::kMode != kV2V && Config::kMode != kV2E
        && Config::kMode != kV2N) {
      LOG(FATAL) << "Advance from edge not supported for CPU";
    }
    IntArray1D lcl_row_offsets;
    mg_int out_len = 0;
    if (Config::kAdvanceAll) {
      lcl_row_offsets = csr.row_offsets;
      out_len = csr.column_indices.length;
    } else {
      if (Config::kMode != kV2N && Config::kMode != kE2N) {
        lcl_row_offsets.length = input_frontier.length + 1;
        lcl_row_offsets.data = alloc->template AllocateWorkspace<mg_int>(
            lcl_row_offsets.length * sizeof(mg_int));
        out_len = ComputeOutputLength<Alloc>(
            csr, input_frontier, &lcl_row_offsets, alloc);
      }
    }
    if (output_frontier) {
      if (output_frontier->data == nullptr) {
        // The output frontier buffer should be allocated.
        output_frontier->length = out_len;
        output_frontier->data = alloc->template AllocateData<mg_int>(
            output_frontier->length * sizeof(mg_int));
      } else {
        CHECK_GE(output_frontier->length, out_len)
          << "Require output frontier of length " << out_len
          << " but only got a buffer of length " << output_frontier->length;
      }
    }

    IntArray1D outbuf = (output_frontier)? *output_frontier : IntArray1D();
    CPUAdvance<Config, GData, Functor, Alloc>(
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
