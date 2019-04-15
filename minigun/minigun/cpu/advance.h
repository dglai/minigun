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

template <typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
class CPUAdvanceExecutor {
 public:
  CPUAdvanceExecutor(
      Csr csr,  // pass by value to make sure it is copied to device memory
      GData* gdata,
      IntArray1D input_frontier,
      IntArray1D output_frontier,
      Alloc* alloc):
    csr_(csr), gdata_(gdata),
    input_frontier_(input_frontier), output_frontier_(output_frontier),
    alloc_(alloc) {}

  void Run() {
    IntArray1D lcl_row_offsets;
    if (Config::kAdvanceAll) {
      lcl_row_offsets = csr_.row_offsets;
      output_frontier_.length = csr_.column_indices.length;
    } else {
      lcl_row_offsets.length = input_frontier_.length + 1;
      lcl_row_offsets.data = alloc_->template AllocateWorkspace<mg_int>(
          lcl_row_offsets.length * sizeof(mg_int));
      ComputeOutputLength(&lcl_row_offsets, &output_frontier_.length);
    }

    if (Config::kMode != kV2N && Config::kMode != kE2N &&
        output_frontier_.data == nullptr) {
      output_frontier_.data = alloc_->template AllocateData<mg_int>(
          output_frontier_.length * sizeof(mg_int));
    }

    CPUAdvance(lcl_row_offsets);

    if (!Config::kAdvanceAll) {
      alloc_->FreeWorkspace(lcl_row_offsets.data);
    }
  }

  void ComputeOutputLength(IntArray1D* lcl_row_offsets, mg_int* outlen) {
    IntArray1D edge_counts;
    lcl_row_offsets->data[0] = 0;
    edge_counts.length = input_frontier_.length;
    edge_counts.data = alloc_->template AllocateWorkspace<mg_int>(
        edge_counts.length * sizeof(mg_int));
#pragma omp parallel for
    for (mg_int tid = 0; tid < edge_counts.length; ++tid) {
      const mg_int vid = input_frontier_.data[tid];
      edge_counts.data[tid] = csr_.row_offsets.data[vid + 1] -
        csr_.row_offsets.data[vid];
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
        thread_sum = alloc_->template AllocateWorkspace<mg_int>(ntr *
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

    alloc_->FreeWorkspace(edge_counts.data);
    alloc_->FreeWorkspace(thread_sum);
    *outlen = lcl_row_offsets->data[edge_counts.length];
  }

  void CPUAdvance(IntArray1D lcl_row_offsets) {
    const mg_int M = output_frontier_.length;
    IntArray1D lcl2glb;
    if (!Config::kAdvanceAll) {
      lcl2glb.length = lcl_row_offsets.length - 1;
      lcl2glb.data = alloc_->template AllocateWorkspace<mg_int>(sizeof(mg_int)
          * lcl2glb.length);
#pragma omp parallel for
      for (mg_int in_item = 0; in_item < lcl2glb.length; ++in_item) {
        mg_int src = 0;
        if (Config::kMode == kE2V || Config::kMode == kE2E) {
          if (Config::kAdvanceAll) {
            src = csr_.column_indices.data[in_item];
          } else {
            src = csr_.column_indices.data[input_frontier_.data[in_item]];
          }
        } else {
          if (Config::kAdvanceAll) {
            src = in_item;
          } else {
            src = input_frontier_.data[in_item];
          }
        }
        lcl2glb.data[in_item] = src;
      }
    }

#pragma omp parallel for
    for (mg_int out_item = 0; out_item < M; ++out_item) {
      // find src, dst, and eid
      mg_int src, dst, eid;
      if (Config::kAdvanceAll) {
        src = CPUBinarySearchSrc(csr_.row_offsets, out_item);
        eid = out_item;
        dst = csr_.column_indices.data[eid];
      } else {
        mg_int lclsrc = CPUBinarySearchSrc(lcl_row_offsets, out_item);
        src = lcl2glb.data[lclsrc];
        eid = csr_.row_offsets.data[src] + (out_item -
            lcl_row_offsets.data[lclsrc]);
        dst = csr_.column_indices.data[eid];
      }

      if (Functor::CondEdge(src, dst, eid, gdata_)) {
        Functor::ApplyEdge(src, dst, eid, gdata_);
        if (Config::kMode == kV2V || Config::kMode == kE2V) {
          output_frontier_.data[out_item] = dst;
        } else if (Config::kMode == kV2E || Config::kMode == kE2E) {
          output_frontier_.data[out_item] = eid;
        }
      } else {
        if (Config::kMode != kV2N && Config::kMode != kE2N) {
          output_frontier_.data[out_item] = kInvalid;
        }
      }
    }

    if (!Config::kAdvanceAll) {
      alloc_->FreeWorkspace(lcl2glb.data);
    }
  }

 private:
  const Csr& csr_;
  GData* gdata_;
  IntArray1D input_frontier_;
  IntArray1D output_frontier_;
  Alloc* alloc_;
};


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
      IntArray1D output_frontier,
      Alloc* alloc) {
    CPUAdvanceExecutor<Config, GData, Functor, Alloc> exec(
        csr, gdata, input_frontier, output_frontier, alloc);
    exec.Run();
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CPU_ADVANCE_H_
