#ifndef MINIGUN_CUDA_ADVANCE_LB_CUH_
#define MINIGUN_CUDA_ADVANCE_LB_CUH_

#include "./cuda_common.cuh"

namespace minigun {
namespace advance {

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
          typename Config,
          typename GData,
          typename Functor>
__global__ void CUDAAdvanceLBKernel(
    Csr csr,
    GData gdata,
    IntArray1D input_frontier,
    IntArray1D output_frontier,
    IntArray1D lcl_row_offsets,
    int nparts_per_blk,
    IntArray1D partition_starts) {

  // XXX: currently N_SMEM_ELEMENTS must be equal to blockDim.y
  __shared__ mg_int s_lcl_row_offsets[N_SMEM_ELEMENTS];
  __shared__ mg_int s_glb_row_offsets[N_SMEM_ELEMENTS];
  __shared__ mg_int s_lcl2glb_vid[N_SMEM_ELEMENTS];

  mg_int blk_out_start = blockDim.y * nparts_per_blk * blockIdx.y;
  mg_int part_idx = blockIdx.y * nparts_per_blk;
  const mg_int loop_end = min(partition_starts.length - 1,
                              part_idx + nparts_per_blk);
  while (part_idx < loop_end) {
    // cooperatively load row offsets into load shared mem
    // each thread is in charge of one vertex
    // TODO(minjie): can use more threads
    const mg_int part_start = max(_ldg(partition_starts.data + part_idx) - 1,
        static_cast<mg_int>(0));
    const mg_int part_end = _ldg(partition_starts.data + part_idx + 1);
    const mg_int in_item = part_start + threadIdx.y;
    //printf("pidx=%ld, st=%ld ed=%ld\n", part_idx, part_start, part_end);
    if (in_item < part_end) {
      s_lcl_row_offsets[threadIdx.y] = _ldg(lcl_row_offsets.data + in_item);
      mg_int src = 0;
      if (Config::kMode == kE2V || Config::kMode == kE2E) {
        if (Config::kAdvanceAll) {
          src = _ldg(csr.column_indices.data + in_item);
        } else {
          const mg_int in_eid = _ldg(input_frontier.data + in_item);
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
      s_lcl_row_offsets[threadIdx.y] = types::MaxValue<mg_int>();
      s_glb_row_offsets[threadIdx.y] = kInvalid;
      s_lcl2glb_vid[threadIdx.y] = kInvalid;
    }
    __syncthreads();

    // cooperatively process edges mapped by the row offsets
    // in the shared memory; each thread is in charge of one edge
    const mg_int out_item = blk_out_start + threadIdx.y;
    //printf("(%d, %d): pidx=%ld %ld\n", blockIdx.y, threadIdx.y, part_idx, out_item);
    if (out_item < output_frontier.length) {
      // TODO(minjie): binary search is not always needed
      const mg_int s_lclsrc = BinarySearch<N_SMEM_ELEMENTS>(out_item, s_lcl_row_offsets) - 1;
      const mg_int src = s_lcl2glb_vid[s_lclsrc];
      // find the index of the current edge w.r.t. its src node
      const mg_int veid = out_item - s_lcl_row_offsets[s_lclsrc];
      const mg_int eid = s_glb_row_offsets[s_lclsrc] + veid;
      const mg_int dst = _ldg(csr.column_indices.data + eid);
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
          output_frontier.data[out_item] = kInvalid;
        }
      }
    }
    __syncthreads();

    part_idx += 1;
    blk_out_start += blockDim.y;
  }
}

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_LB_CUH_
