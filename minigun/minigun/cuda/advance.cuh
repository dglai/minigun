#ifndef MINIGUN_CUDA_ADVANCE_CUH_
#define MINIGUN_CUDA_ADVANCE_CUH_

#include <sstream>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/transform.hxx>

#include "../advance.h"
#include "./advance_all.cuh"
#include "./advance_lb.cuh"
#include "./cuda_common.cuh"
#include "./tuning.h"

namespace minigun {
namespace advance {

#define MAX_NTHREADS 1024
#define MAX_NBLOCKS 65535L

__inline__ void PrintDev(IntArray1D arr) {
  mg_int* tmp = new mg_int[arr.length];
  CUDA_CALL(cudaMemcpy(tmp, arr.data, sizeof(mg_int) * arr.length, cudaMemcpyDeviceToHost));
  std::ostringstream oss;
  oss << "[";
  for (mg_int i = 0; i < arr.length; ++i) {
    oss << tmp[i] << ", ";
  }
  oss << "]";
  LOG(INFO) << oss.str();
  delete [] tmp;
}

struct StridedIterator :
  mgpu::const_iterator_t<StridedIterator, int, mg_int> {

  StridedIterator() = default;
  MGPU_HOST_DEVICE StridedIterator(mg_int offset, mg_int stride, mg_int bound) :
    mgpu::const_iterator_t<StridedIterator, int, mg_int>(0),
    offset_(offset), stride_(stride), bound_(bound) { }

  MGPU_HOST_DEVICE mg_int operator()(int index) const {
    mg_int ret = offset_ + index * stride_;
    return (ret < bound_) ? ret : bound_;
  }

  mg_int offset_, stride_, bound_;
};

template <typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
class CudaAdvanceExecutor {
 public:
  CudaAdvanceExecutor(
      AdvanceAlg algo,
      const RuntimeConfig& rtcfg,
      const Csr& csr,
      GData* gdata,
      IntArray1D input_frontier,
      IntArray1D output_frontier,
      Alloc* alloc):
    algo_(algo), rtcfg_(rtcfg), csr_(csr), gdata_(gdata),
    input_frontier_(input_frontier), output_frontier_(output_frontier),
    alloc_(alloc) { }

  void Run() {
    MgpuContext<Alloc> mgpuctx(rtcfg_.stream, alloc_);
    // Row offset array of local graph (graph sliced by nodes in input_frontier).
    // It's length == len(input_frontier) + 1, and
    // lcl_row_offsets[i+1] - lcl_row_offsets[i] == edge_counts[i]
    IntArray1D lcl_row_offsets;
    lcl_row_offsets.length = input_frontier_.length + 1;
    lcl_row_offsets.data = alloc_->template AllocateWorkspace<mg_int>(
        lcl_row_offsets.length * sizeof(mg_int));
    ComputeOutputLength(mgpuctx, &lcl_row_offsets, &output_frontier_.length);
  
    if (Config::kMode != kV2N && Config::kMode != kE2N
        && output_frontier_.data == nullptr) {
      // The output frontier buffer should be allocated.
      output_frontier_.data = alloc_->template AllocateData<mg_int>(
          output_frontier_.length * sizeof(mg_int));
    }

    switch (algo_) {
      case kGunrockLBOut :
        //CudaAdvanceGunrockLBOut(mgpuctx, lcl_row_offsets);
        break;
      default:
        LOG(FATAL) << "Algorithm " << algo_ << " is not supported.";
    }

    if (lcl_row_offsets.data) {
      alloc_->FreeWorkspace(lcl_row_offsets.data);
    }
  }

  void ComputeOutputLength(MgpuContext<Alloc>& mgpuctx,
      IntArray1D* lcl_row_offsets, mg_int* outlen) {
    // An array of size len(input_frontier). Each element i is the number of
    // edges input_frointer[i] has.
    IntArray1D edge_counts;
    edge_counts.length = input_frontier_.length;
    edge_counts.data = alloc_->template AllocateWorkspace<mg_int>(
        edge_counts.length * sizeof(mg_int));
    mgpu::transform(
      [] MGPU_DEVICE(int tid, const mg_int* rowoff, const mg_int* infront, mg_int* ecounts) {
        const mg_int vid = infront[tid];
        ecounts[tid] = rowoff[vid + 1] - rowoff[vid];
      }, edge_counts.length, mgpuctx,
      csr_.row_offsets.data,
      input_frontier_.data,
      edge_counts.data);
    // compute output len using prefix scan
    mgpu::scan<mgpu::scan_type_exc>(
        edge_counts.data,
        edge_counts.length,
        lcl_row_offsets->data,
        mgpu::plus_t<mg_int>(),
        lcl_row_offsets->data + edge_counts.length,  // the last element stores the reduction.
        mgpuctx);
    // get the reduction
    CUDA_CALL(cudaMemcpy(outlen, lcl_row_offsets->data + edge_counts.length,
          sizeof(mg_int), cudaMemcpyDeviceToHost));
    LOG(INFO) << "Output frontier length: " << *outlen;
    alloc_->FreeWorkspace(edge_counts.data);
  }

  void CudaAdvanceGunrockLBOut(MgpuContext<Alloc>& mgpuctx, IntArray1D lcl_row_offsets) {
    // partition the workload
    const mg_int M = output_frontier_.length;
    const int ty = MAX_NTHREADS / rtcfg_.data_num_threads;
    const int ny = ty * 2;  // XXX: each block handles two partitions
    const int by = std::min((M + ny - 1) / ny, MAX_NBLOCKS);
    const int nparts_per_blk = ((M + by - 1) / by + ty - 1) / ty;
    const dim3 nblks(rtcfg_.data_num_blocks, by);
    const dim3 nthrs(rtcfg_.data_num_threads, ty);
    LOG(INFO) << "Blocks: (" << nblks.x << "," << nblks.y << ") Threads: ("
      << nthrs.x << "," << nthrs.y << ")" << " nparts_per_blk=" << nparts_per_blk;
    // use sorted search to compute the partition_starts 
    IntArray1D partition_starts;
    partition_starts.length = (M + ty - 1) / ty + 1;
    partition_starts.data = alloc_->template AllocateWorkspace<mg_int>(
        partition_starts.length * sizeof(mg_int));
    mgpu::sorted_search<mgpu::bounds_lower>(
        StridedIterator(0, ty, output_frontier_.length),
        partition_starts.length,
        lcl_row_offsets.data,
        lcl_row_offsets.length,
        partition_starts.data,
        mgpu::less_t<mg_int>(),
        mgpuctx);
    //PrintDev(lcl_row_offsets);
    //PrintDev(partition_starts);

    if (ty > 512) {
      CUDAAdvanceLBKernel<1024, Config, GData, Functor>
        <<<nblks, nthrs, 0, rtcfg_.stream>>>(
            csr_, gdata_, input_frontier_, output_frontier_,
            lcl_row_offsets, nparts_per_blk, partition_starts);
    } else if (ty > 256) {
      CUDAAdvanceLBKernel<512, Config, GData, Functor>
        <<<nblks, nthrs, 0, rtcfg_.stream>>>(
            csr_, gdata_, input_frontier_, output_frontier_,
            lcl_row_offsets, nparts_per_blk, partition_starts);
    } else {
      CUDAAdvanceLBKernel<256, Config, GData, Functor>
        <<<nblks, nthrs, 0, rtcfg_.stream>>>(
            csr_, gdata_, input_frontier_, output_frontier_,
            lcl_row_offsets, nparts_per_blk, partition_starts);
    }

    alloc_->FreeWorkspace(partition_starts.data);
  }

 private:
  const AdvanceAlg algo_;
  const RuntimeConfig& rtcfg_;
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
struct DispatchXPU<kDLGPU, Config, GData, Functor, Alloc> {
  static void Advance(
      const RuntimeConfig& rtcfg,
      const Csr& csr,
      GData* gdata,
      IntArray1D input_frontier,
      IntArray1D output_frontier,
      Alloc* alloc) {
    // Call advance
    if (Config::kAdvanceAll) {
      AdvanceAlg algo = FindAdvanceAllAlgo<Config>(rtcfg, csr);
      CudaAdvanceAll<Config, GData, Functor, Alloc>(
          algo, rtcfg, csr, gdata, output_frontier, alloc);
    } else {
      AdvanceAlg algo = FindAdvanceAlgo<Config>(rtcfg, csr,
          input_frontier, output_frontier);
      CudaAdvanceExecutor<Config, GData, Functor, Alloc> exec(
          algo, rtcfg, csr, gdata, input_frontier, output_frontier, alloc);
      exec.Run();
    }
  }
};

#undef MAX_NTHREADS
#undef MAX_NBLOCKS

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_CUH_
