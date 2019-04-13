#ifndef MINIGUN_CUDA_ADVANCE_LB_CUH_
#define MINIGUN_CUDA_ADVANCE_LB_CUH_

namespace minigun {
namespace advance {

/*
 * lcl_row_offsets:
 *  - length == len(input_frontier)
 *  - row offset of the subgraph formed by input_frontier
 * num_partitions: 
 *  - number of edge partitions handled by one block
 * row_offset_partitions:
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
    GData* gdata,
    IntArray1D input_frontier,
    IntArray1D output_frontier,
    IntArray1D lcl_row_offsets,
    int num_partitions,
    IntArray1D row_offset_partitions) {

  // XXX: currently N_SMEM_ELEMENTS must be equal to blockDim.y
  __shared__ mg_int s_lcl_row_offsets[N_SMEM_ELEMENTS];
  __shared__ mg_int s_glb_row_offsets[N_SMEM_ELEMENTS];
  __shared__ mg_int s_lcl2glb_vid[N_SMEM_ELEMENTS];

  mg_int blk_out_start = 0;
  mg_int part_idx = blockIdx.y;
  while (part_idx < row_offset_partitions.length - 1) {
    const mg_int part_start = row_offset_partitions.data[part_idx];
    const mg_int part_end = row_offset_partitions.data[part_idx + 1];
    // cooperatively load row offsets into load shared mem
    // each thread is in charge of one vertex
    // INVARIANT: (part_end - part_start) <= blockDim.y
    const mg_int in_item = part_start + threadIdx.y;
    if (in_item < part_end) {
      s_lcl_row_offsets[threadIdx.y] = lcl_row_offsets[in_item];
      const mg_int src = (Config::kMode == kE2V || Config::kMode == kE2E)?
        csr.column_indices[input_frontier.data[in_item]] : input_frontier.data[in_item];
      s_glb_row_offsets[threadIdx.y] = csr.row_offsets[src];
      s_lcl2glb_vid[threadIdx.y] = src;
    }
    __syncthreads();

    // cooperatively process edges mapped by the row offsets
    // in the shared memory; each thread is in charge of one edge
    const mg_int out_item = blk_out_start + threadIdx.y;
    if (out_item < output_frontier.length) {
      // TODO(minjie): binary search is not always needed
      const mg_int s_lclsrc = BinarySearchSrc<N_SMEM_ELEMENTS>(s_lcl_row_offsets, threadIdx.y);
      const mg_int src = s_lcl2glb_vid[s_lclsrc];
      // find the index of the current edge w.r.t. its src node
      const mg_int veid = threadIdx.y - (s_lclsrc == 0)? 0 : s_lcl_row_offsets[s_lclsrc - 1];
      const mg_int eid = s_glb_row_offsets[s_lclsrc] + veid;
      const mg_int dst = csr.column_indices[eid];
      if (Functor::CondEdge(src, dst, eid, gdata)) {
        Functor::ApplyEdge(src, dst, eid, gdata);
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
