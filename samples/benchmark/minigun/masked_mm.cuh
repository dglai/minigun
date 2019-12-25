#ifndef SAMPLES_BENCHMARK_MINIGUN_MASKED_MM_CUH_
#define SAMPLES_BENCHMARK_MINIGUN_MASKED_MM_CUH_

#include <cuda_runtime.h>
#include "../../samples_io.h"
#include "../../samples_utils.h"

namespace masked_mm {

struct GData {
  int D = 0;  // feat size
  int H = 0;  // num heads
  float* ndata{nullptr};
  float* score{nullptr};
};

struct MaskedMMFunctor {
  static __device__ __forceinline__ bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    const int D = gdata->D;
    const int H = gdata->H;
    // each thread handles one attention head
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    while (h < H) {
      const float* srcoff = gdata->ndata + (src * H + h) * D;
      const float* dstoff = gdata->ndata + (dst * H + h) * D;
      float sum = 0.;
      for (int i = 0; i < D; ++i) {
        sum += __ldg(srcoff + i) * __ldg(dstoff + i);
      }
      gdata->score[eid * H + h] = sum;
      h += blockDim.x;
    }
  }
  static __device__ __forceinline__ void ApplyEdgeReduce(
    int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float* val, GData* gdata) {}
  static __device__ __forceinline__ int32_t GetFeatSize(GData *gdata) {
    return -1;
  }
  static __device__ __forceinline__ float* GetOutBuf(GData* gdata) {
    return nullptr;
  }
  static __device__ __forceinline__ int32_t GetOutOffset(int32_t idx, GData* gdata) {
    return idx;
  }
};

void InitGData(const utils::SampleCsr& csr, GData* gdata, GData* truth) {
  const int32_t N = csr.row_offsets.size() - 1;
  const int32_t M = csr.column_indices.size();
  const int H = gdata->H, D = gdata->D;
  std::vector<float> ndata(N * gdata->D * gdata->H), score(M * gdata->H, 0.);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (float)rand() / RAND_MAX;
  }
  CUDA_CALL(cudaMalloc(&(gdata->ndata), sizeof(float) * N * gdata->D * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->ndata, &ndata[0],
        sizeof(float) * N * gdata->D * gdata->H, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->score), sizeof(float) * M * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->score, &score[0],
        sizeof(float) * M * gdata->H, cudaMemcpyHostToDevice));
  // compute truth
  truth->score = new float[gdata->H * M];
  for (size_t u = 0; u < csr.row_offsets.size() - 1; u++) {
    for (int32_t eid = csr.row_offsets[u]; eid < csr.row_offsets[u + 1]; eid++) {
      int32_t v = csr.column_indices[eid];
      for (int32_t idx = 0; idx < H * D; idx++)
        truth->score[eid * H + idx / D] += ndata[u * H * D + idx] * ndata[v * H * D + idx];
    }
  }
}

void FreeGData(GData* gdata, GData* truth) {
  CUDA_CALL(cudaFree(gdata->ndata));
  CUDA_CALL(cudaFree(gdata->score));
  delete [] truth->score;
}

void CheckResult(const utils::SampleCsr& csr, GData* gdata, GData* truth) {
  const int32_t M = csr.column_indices.size();
  const int H = gdata->H;
  float* h_score = new float[M * H];
  CUDA_CALL(cudaMemcpy(h_score, gdata->score, sizeof(float) * M * H, cudaMemcpyDeviceToHost));
  bool equal = utils::IterEqual(h_score, truth->score, M * H);
  std::cout << "Correct? " << equal << std::endl;
  delete [] h_score;
}

}  // masked_mm

#endif
