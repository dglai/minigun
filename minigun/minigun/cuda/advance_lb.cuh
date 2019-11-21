#ifndef MINIGUN_CUDA_ADVANCE_LB_CUH_
#define MINIGUN_CUDA_ADVANCE_LB_CUH_

#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/transform.hxx>

#include "./cuda_common.cuh"

namespace minigun {

// Cuda context that is compatible with modern gpu
template <typename Alloc>
class MgpuContext : public mgpu::context_t {
 public:
  MgpuContext(int device_id, cudaStream_t stream, Alloc* alloc):
    cuda_ctx_(CudaContext::Get(device_id)),
    stream_(stream), alloc_(alloc) {
    //CUDA_CALL(cudaEventCreate(&event_));
  }
  ~MgpuContext() {
    //CUDA_CALL(cudaEventDestroy(event_));
  }
  const cudaDeviceProp& props() const override {
    return cuda_ctx_.props();
  } 
  int ptx_version() const override {
    return cuda_ctx_.ptx_version();
  }
  cudaStream_t stream() override {
    return stream_;
  }
  void* alloc(size_t size, mgpu::memory_space_t space) override {
    CHECK_EQ(space,  mgpu::memory_space_device);
    return alloc_->template AllocateWorkspace<void>(size);
  }
  void free(void* p, mgpu::memory_space_t space) override {
    CHECK_EQ(space,  mgpu::memory_space_device);
    alloc_->FreeWorkspace(p);
  }
  void synchronize() override {
    if (stream_) {
      CUDA_CALL(cudaStreamSynchronize(stream_));
    } else {
      CUDA_CALL(cudaDeviceSynchronize());
    }
  }
  cudaEvent_t event() override {
    LOG(FATAL) << "event is not implemented.";
    return event_;
  }
  void timer_begin() override {
    LOG(FATAL) << "timer_begin is not implemented.";
  }
  double timer_end() override {
    LOG(FATAL) << "timer_end is not implemented.";
    return 0.0;
  }

 private:
  const CudaContext& cuda_ctx_;
  cudaStream_t stream_;
  Alloc* alloc_;
  cudaEvent_t event_;
};

namespace advance {

#define MAX_NTHREADS 1024
#define MAX_NBLOCKS 65535

// From Andrew Davidson's dStepping SSSP GPU implementation
// binary search on device, only works for arrays shorter
// than 1024
// Return the closest element that is greater than the key.
template <int NT, typename KeyType, typename ArrayType>
__device__ int BinarySearch(KeyType i, ArrayType* queue) {
  int mid = ((NT >> 1) - 1);

  if (NT > 512) mid = queue[mid] > i ? mid - 256 : mid + 256;
  if (NT > 256) mid = queue[mid] > i ? mid - 128 : mid + 128;
  if (NT > 128) mid = queue[mid] > i ? mid - 64 : mid + 64;
  if (NT > 64) mid = queue[mid] > i ? mid - 32 : mid + 32;
  if (NT > 32) mid = queue[mid] > i ? mid - 16 : mid + 16;
  mid = queue[mid] > i ? mid - 8 : mid + 8;
  mid = queue[mid] > i ? mid - 4 : mid + 4;
  mid = queue[mid] > i ? mid - 2 : mid + 2;
  mid = queue[mid] > i ? mid - 1 : mid + 1;
  mid = queue[mid] > i ? mid : mid + 1;

  return mid;
}

/*
 * lcl_row_offsets:
 *  - length == len(input_frontier)
 *  - row offset of the subgraph formed by input_frontier
 * nparts_per_blk: 
 *  - number of edge partitions handled by one block
 * partition_starts:
 *  - stores the start position of the rowoffset for each edge partition
 *  - length == total number of partitions + 1
 *  - start position of each partition
 */
template <int N_SMEM_ELEMENTS,
          typename Idx,
          typename Config,
          typename GData,
          typename Functor>
__global__ void CUDAAdvanceLBKernel(
    Csr<Idx> csr,
    GData gdata,
    IntArray1D<Idx> input_frontier,
    IntArray1D<Idx> output_frontier,
    IntArray1D<Idx> lcl_row_offsets,
    int nparts_per_blk,
    IntArray1D<Idx> partition_starts) {

  // XXX: currently N_SMEM_ELEMENTS must be equal to blockDim.y
  __shared__ Idx s_lcl_row_offsets[N_SMEM_ELEMENTS];
  __shared__ Idx s_glb_row_offsets[N_SMEM_ELEMENTS];
  __shared__ Idx s_lcl2glb_vid[N_SMEM_ELEMENTS];

  Idx blk_out_start = blockDim.y * nparts_per_blk * blockIdx.y;
  Idx part_idx = blockIdx.y * nparts_per_blk;
  const Idx loop_end = min(partition_starts.length - 1, part_idx + nparts_per_blk);
  while (part_idx < loop_end) {
    // cooperatively load row offsets into load shared mem
    // each thread is in charge of one vertex
    // TODO(minjie): can use more threads
    const Idx part_start = max(_ldg(partition_starts.data + part_idx) - 1,
        static_cast<Idx>(0));
    const Idx part_end = _ldg(partition_starts.data + part_idx + 1);
    const Idx in_item = part_start + threadIdx.y;
    //printf("pidx=%ld, st=%ld ed=%ld\n", part_idx, part_start, part_end);
    if (in_item < part_end) {
      s_lcl_row_offsets[threadIdx.y] = _ldg(lcl_row_offsets.data + in_item);
      Idx src = 0;
      if (Config::kMode == kE2V || Config::kMode == kE2E) {
        if (Config::kAdvanceAll) {
          src = _ldg(csr.column_indices.data + in_item);
        } else {
          const Idx in_eid = _ldg(input_frontier.data + in_item);
          src = _ldg(csr.column_indices.data + in_eid);
        }
      } else {
        if (Config::kAdvanceAll) {
          src = in_item;
        } else {
          src = _ldg(input_frontier.data + in_item);
        }
      }
      s_glb_row_offsets[threadIdx.y] = _ldg(csr.row_offsets.data + src);
      s_lcl2glb_vid[threadIdx.y] = src;
    } else {
      s_lcl_row_offsets[threadIdx.y] = types::MaxValue<Idx>();
      s_glb_row_offsets[threadIdx.y] = MG_INVALID;
      s_lcl2glb_vid[threadIdx.y] = MG_INVALID;
    }
    __syncthreads();

    // cooperatively process edges mapped by the row offsets
    // in the shared memory; each thread is in charge of one edge
    const Idx out_item = blk_out_start + threadIdx.y;
    //printf("(%d, %d): pidx=%ld %ld\n", blockIdx.y, threadIdx.y, part_idx, out_item);
    if (out_item < output_frontier.length) {
      // TODO(minjie): binary search is not always needed
      const Idx s_lclsrc = BinarySearch<N_SMEM_ELEMENTS>(out_item, s_lcl_row_offsets) - 1;
      const Idx src = s_lcl2glb_vid[s_lclsrc];
      // find the index of the current edge w.r.t. its src node
      const Idx veid = out_item - s_lcl_row_offsets[s_lclsrc];
      const Idx eid = s_glb_row_offsets[s_lclsrc] + veid;
      const Idx dst = _ldg(csr.column_indices.data + eid);
      //printf("%ld %ld %ld %ld %ld\n", out_item, s_lclsrc, src, eid, dst);
      if (Functor::CondEdge(src, dst, eid, &gdata)) {
        Functor::ApplyEdge(src, dst, eid, &gdata);
        // Add dst/eid to output frontier
        if (Config::kMode == kV2V || Config::kMode == kE2V) {
          output_frontier.data[out_item] = dst;
        } else if (Config::kMode == kV2E || Config::kMode == kE2E) {
          output_frontier.data[out_item] = eid;
        }
      } else {
        if (Config::kMode != kV2N && Config::kMode != kE2N) {
          // Add invalid to output frontier
          output_frontier.data[out_item] = MG_INVALID;
        }
      }
    }
    __syncthreads();

    part_idx += 1;
    blk_out_start += blockDim.y;
  }
}

template <typename Idx>
struct StridedIterator :
  mgpu::const_iterator_t<StridedIterator<Idx>, int, Idx> {

  StridedIterator() = default;
  MGPU_HOST_DEVICE StridedIterator(Idx offset, Idx stride, Idx bound) :
    mgpu::const_iterator_t<StridedIterator, int, Idx>(0),
    offset_(offset), stride_(stride), bound_(bound) { }

  MGPU_HOST_DEVICE Idx operator()(int index) const {
    Idx ret = offset_ + index * stride_;
    return (ret < bound_) ? ret : bound_;
  }

  Idx offset_, stride_, bound_;
};

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
class CudaAdvanceExecutor {
 public:
  CudaAdvanceExecutor(
      AdvanceAlg algo,
      const RuntimeConfig& rtcfg,
      const Csr<Idx>& csr,
      GData* gdata,
      IntArray1D<Idx> input_frontier,
      IntArray1D<Idx>* output_frontier,
      Alloc* alloc):
    algo_(algo), rtcfg_(rtcfg), csr_(csr), gdata_(gdata),
    input_frontier_(input_frontier), output_frontier_(output_frontier),
    alloc_(alloc) { }

  void Run() {
    MgpuContext<Alloc> mgpuctx(rtcfg_.ctx.device_id, rtcfg_.stream, alloc_);
    // Row offset array of local graph (graph sliced by nodes in input_frontier).
    // It's length == len(input_frontier) + 1, and
    // lcl_row_offsets[i+1] - lcl_row_offsets[i] == edge_counts[i]
    IntArray1D<Idx> lcl_row_offsets;

    if (Config::kAdvanceAll) {
      lcl_row_offsets = csr_.row_offsets;
      out_len_ = csr_.column_indices.length;
    } else {
      lcl_row_offsets.length = input_frontier_.length + 1;
      lcl_row_offsets.data = alloc_->template AllocateWorkspace<Idx>(
          lcl_row_offsets.length * sizeof(Idx));
      out_len_ = ComputeOutputLength(mgpuctx, &lcl_row_offsets);
    }

    if (output_frontier_) {
      if (output_frontier_->data == nullptr) {
        // The output frontier buffer should be allocated.
        output_frontier_->length = out_len_;
        output_frontier_->data = alloc_->template AllocateData<Idx>(
            output_frontier_->length * sizeof(Idx));
      } else {
        CHECK_GE(output_frontier_->length, out_len_)
          << "Require output frontier of length " << out_len_
          << " but only got a buffer of length " << output_frontier_->length;
      }
    }

    switch (algo_) {
      case kGunrockLBOut :
        CudaAdvanceGunrockLBOut(mgpuctx, lcl_row_offsets);
        break;
      default:
        LOG(FATAL) << "Algorithm " << algo_ << " is not supported.";
    }

    if (!Config::kAdvanceAll) {
      alloc_->FreeWorkspace(lcl_row_offsets.data);
    }
  }

  Idx ComputeOutputLength(MgpuContext<Alloc>& mgpuctx, IntArray1D<Idx>* lcl_row_offsets) {
    // An array of size len(input_frontier). Each element i is the number of
    // edges input_frointer[i] has.
    IntArray1D<Idx> edge_counts;
    edge_counts.length = input_frontier_.length;
    edge_counts.data = alloc_->template AllocateWorkspace<Idx>(
        edge_counts.length * sizeof(Idx));
    mgpu::transform(
      [] MGPU_DEVICE(int tid, const Idx* rowoff, const Idx* infront, Idx* ecounts) {
        const Idx vid = infront[tid];
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
        mgpu::plus_t<Idx>(),
        lcl_row_offsets->data + edge_counts.length,  // the last element stores the reduction.
        mgpuctx);
    //// get the reduction
    Idx outlen;
    CUDA_CALL(cudaMemcpy(&outlen, lcl_row_offsets->data + edge_counts.length,
          sizeof(Idx), cudaMemcpyDeviceToHost));
    //LOG(INFO) << "Output frontier length: " << *outlen;
    alloc_->FreeWorkspace(edge_counts.data);
    return outlen;
  }

  void CudaAdvanceGunrockLBOut(MgpuContext<Alloc>& mgpuctx, IntArray1D<Idx> lcl_row_offsets) {
    // partition the workload
    const Idx M = out_len_;
    const int ty = MAX_NTHREADS / rtcfg_.data_num_threads;
    const int ny = ty * 2;  // XXX: each block handles two partitions
    const int by = std::min((M + ny - 1) / ny, static_cast<Idx>(MAX_NBLOCKS));
    const int nparts_per_blk = ((M + by - 1) / by + ty - 1) / ty;
    const dim3 nblks(rtcfg_.data_num_blocks, by);
    const dim3 nthrs(rtcfg_.data_num_threads, ty);
    //LOG(INFO) << "Blocks: (" << nblks.x << "," << nblks.y << ") Threads: ("
    //  << nthrs.x << "," << nthrs.y << ")" << " nparts_per_blk=" << nparts_per_blk;
    // use sorted search to compute the partition_starts 
    IntArray1D<Idx> partition_starts;
    partition_starts.length = (M + ty - 1) / ty + 1;
    partition_starts.data = alloc_->template AllocateWorkspace<Idx>(
        partition_starts.length * sizeof(Idx));
    mgpu::sorted_search<mgpu::bounds_lower>(
        StridedIterator<Idx>(0, ty, out_len_),
        partition_starts.length,
        lcl_row_offsets.data,
        lcl_row_offsets.length,
        partition_starts.data,
        mgpu::less_t<Idx>(),
        mgpuctx);

    IntArray1D<Idx> outbuf = (output_frontier_)? *output_frontier_ : IntArray1D<Idx>();
    if (ty > 512) {
      CUDAAdvanceLBKernel<1024, Idx, Config, GData, Functor>
        <<<nblks, nthrs, 0, rtcfg_.stream>>>(
            csr_, *gdata_, input_frontier_, outbuf,
            lcl_row_offsets, nparts_per_blk, partition_starts);
    } else if (ty > 256) {
      CUDAAdvanceLBKernel<512, Idx, Config, GData, Functor>
        <<<nblks, nthrs, 0, rtcfg_.stream>>>(
            csr_, *gdata_, input_frontier_, outbuf,
            lcl_row_offsets, nparts_per_blk, partition_starts);
    } else if (ty > 128) {
      CUDAAdvanceLBKernel<256, Idx, Config, GData, Functor>
        <<<nblks, nthrs, 0, rtcfg_.stream>>>(
            csr_, *gdata_, input_frontier_, outbuf,
            lcl_row_offsets, nparts_per_blk, partition_starts);
    } else if (ty > 64) {
      CUDAAdvanceLBKernel<128, Idx, Config, GData, Functor>
        <<<nblks, nthrs, 0, rtcfg_.stream>>>(
            csr_, *gdata_, input_frontier_, outbuf,
            lcl_row_offsets, nparts_per_blk, partition_starts);
    } else if (ty > 32) {
      CUDAAdvanceLBKernel<64, Idx, Config, GData, Functor>
        <<<nblks, nthrs, 0, rtcfg_.stream>>>(
            csr_, *gdata_, input_frontier_, outbuf,
            lcl_row_offsets, nparts_per_blk, partition_starts);
    } else {
      CUDAAdvanceLBKernel<32, Idx, Config, GData, Functor>
        <<<nblks, nthrs, 0, rtcfg_.stream>>>(
            csr_, *gdata_, input_frontier_, outbuf,
            lcl_row_offsets, nparts_per_blk, partition_starts);
    }

    alloc_->FreeWorkspace(partition_starts.data);
  }

 private:
  const AdvanceAlg algo_;
  const RuntimeConfig& rtcfg_;
  const Csr<Idx>& csr_;
  GData* gdata_;
  IntArray1D<Idx> input_frontier_;
  IntArray1D<Idx>* output_frontier_;
  Alloc* alloc_;

  // size of the output frontier
  Idx out_len_ = 0;
};

#undef MAX_NTHREADS
#undef MAX_NBLOCKS

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_LB_CUH_
