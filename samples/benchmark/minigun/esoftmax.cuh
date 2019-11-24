#ifndef SAMPLES_BENCHMARK_MINIGUN_ESOFTMAX_CUH_
#define SAMPLES_BENCHMARK_MINIGUN_ESOFTMAX_CUH_

#include <cuda_runtime.h>
#include "../../samples_io.h"
#include "../../samples_utils.h"

namespace esoftmax {

struct GData {
  int H = 0;  // num heads
  float* score{nullptr};
  float* sum{nullptr};
  float* max{nullptr};
  float* ret{nullptr};
  int* eid_mapping{nullptr};
};

// Max
struct EdgeMax {
  static __device__ __forceinline__ bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {}
  static __device__ __forceinline__ void ApplyEdgeReduce(
      int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float& val, GData* gdata) {
    const int H = gdata->H;
    float* inoff = gdata->score + gdata->eid_mapping[eid] * H;
    val = max(val, __ldg(inoff + feat_idx));
  }
  static __device__ __forceinline__ int32_t GetFeatSize(GData *gdata) {
    return gdata->H;
  }
  static __device__ __forceinline__ float* GetOutBuf(GData* gdata) {
    return gdata->max;
  }
};

// minus max, exp and sum
struct MinusMaxExpSum {
  static __device__ __forceinline__ bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {}
  static __device__ __forceinline__ void ApplyEdgeReduce(
      int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float& val, GData* gdata) {
    const int H = gdata->H;
    const float* score_off = gdata->score + gdata->eid_mapping[eid] * H;
    float* ret_off = gdata->ret + gdata->eid_mapping[eid] * H;
    float* max_off = gdata->max + dst * H;
    const float new_score = expf(__ldg(score_off + feat_idx) - __ldg(max_off + feat_idx));
    val += new_score;
    *(ret_off + feat_idx) = new_score;
  }
  static __device__ __forceinline__ int32_t GetFeatSize(GData *gdata) {
    return gdata->H;
  }
  static __device__ __forceinline__ float* GetOutBuf(GData* gdata) {
    return gdata->sum;
  }
};

// norm (node parallel by destinatino)
struct NormByDst {
  static __device__ __forceinline__ bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {}
  static __device__ __forceinline__ void ApplyEdgeReduce(
      int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float& val, GData* gdata) {
    const int H = gdata->H;
    float* ret_off = gdata->ret + gdata->eid_mapping[eid] * H;
    float* sum_off = gdata->sum + dst * H;
    *(ret_off + feat_idx) /= __ldg(sum_off + feat_idx);
  }
  static __device__ __forceinline__ int32_t GetFeatSize(GData *gdata) {
    return gdata->H;
  }
  static __device__ __forceinline__ float* GetOutBuf(GData* gdata) {
    return nullptr;
  }
};

void InitGData(const utils::SampleCsr& csr, const minigun::IntArray eid_mapping,
    GData* gdata, GData* truth) {
  const int32_t N = csr.row_offsets.size() - 1;
  const int32_t M = csr.column_indices.size();
  const int H = gdata->H;
  std::vector<float> sum(N * gdata->H, 0.), max(N * gdata->H, std::numeric_limits<float>::lowest());
  std::vector<float> score(M * gdata->H, 0.), ret(M * gdata->H, 0.);
  for (size_t i = 0; i < score.size(); ++i) {
    score[i] = (float)rand() / RAND_MAX;
  }
  CUDA_CALL(cudaMalloc(&(gdata->sum), sizeof(float) * N * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->sum, &sum[0],
        sizeof(float) * N * gdata->H, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->max), sizeof(float) * N * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->max, &max[0],
        sizeof(float) * N * gdata->H, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->score), sizeof(float) * M * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->score, &score[0],
        sizeof(float) * M * gdata->H, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->ret), sizeof(float) * M * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->ret, &ret[0],
        sizeof(float) * M * gdata->H, cudaMemcpyHostToDevice));
  gdata->eid_mapping = eid_mapping.data;
  // compute truth
  truth->ret = new float[M * H];
  std::vector<float> tmp(N * H, 0.);
  for (size_t i = 0; i < score.size(); ++i) {
    score[i] = std::exp(score[i]);
  }
  for (size_t u = 0; u < csr.row_offsets.size() - 1; ++u) {
    for (int32_t eid = csr.row_offsets[u]; eid < csr.row_offsets[u+1]; ++eid) {
      int32_t v = csr.column_indices[eid];
      for (int32_t idx = 0; idx < H; ++idx) {
        tmp[v * H + idx] += score[eid * H + idx];
      }
    }
  }
  for (size_t eid = 0; eid < csr.column_indices.size(); ++eid) {
    for (int32_t i = 0; i < H; ++i) {
      truth->ret[eid * H + i] = score[eid * H + i] / tmp[csr.column_indices[eid] * H + i];
    }
  }
}

void FreeGData(GData* gdata, GData* truth) {
  CUDA_CALL(cudaFree(gdata->score));
  CUDA_CALL(cudaFree(gdata->sum));
  CUDA_CALL(cudaFree(gdata->max));
  CUDA_CALL(cudaFree(gdata->ret));
  delete [] truth->ret;
}

void CheckResult(const utils::SampleCsr& csr, GData* gdata, GData* truth) {
  const int32_t M = csr.column_indices.size();
  const int H = gdata->H;
  float* h_ret = new float[M * H];
  CUDA_CALL(cudaMemcpy(h_ret, gdata->ret, sizeof(float) * M * H, cudaMemcpyDeviceToHost));
  bool equal = utils::IterEqual(h_ret, truth->ret, M * H);
  std::cout << "Correct? " << equal << std::endl;
  delete [] h_ret;
}

}  // masked_mm

#endif
