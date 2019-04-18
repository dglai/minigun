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
};

__device__ __forceinline__ float MyAtomicMax(float* addr, float val) {
  uint32_t* addr_as_ui = reinterpret_cast<uint32_t*>(addr);
  uint32_t old = *addr_as_ui;
  uint32_t assumed = old;
  do {
    assumed = old;
    old = atomicCAS(addr_as_ui, assumed,
        __float_as_uint(fmax(val, __uint_as_float(old))));
  } while (assumed != old);
  return __uint_as_float(old);
}

// Max
struct EdgeMax {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    const int H = gdata->H;
    float* inoff = gdata->score + eid * H;
    float* outoff = gdata->max + dst * H;
    while (tx < H) {
      MyAtomicMax(outoff + tx, __ldg(inoff + tx));
      tx += stride_x;
    }
  }
};

// minus max, exp and sum
struct MinusMaxExpSum {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    const int H = gdata->H;
    const float* score_off = gdata->score + eid * H;
    float* ret_off = gdata->ret + eid * H;
    float* max_off = gdata->max + dst * H;
    float* sum_off = gdata->sum + dst * H;
    while (tx < H) {
      const float new_score = expf(__ldg(score_off + tx) - __ldg(max_off + tx));
      atomicAdd(sum_off + tx, new_score);
      *(ret_off + tx) = new_score;
      tx += stride_x;
    }
  }
};

// norm
struct Norm {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    const int H = gdata->H;
    float* ret_off = gdata->ret + eid * H;
    float* sum_off = gdata->sum + dst * H;
    while (tx < H) {
      *(ret_off + tx) /= __ldg(sum_off + tx);
      tx += stride_x;
    }
  }
};

void InitGData(const utils::SampleCsr& csr, GData* gdata, GData* truth) {
  const mg_int N = csr.row_offsets.size() - 1;
  const mg_int M = csr.column_indices.size();
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
  // compute truth
  truth->ret = new float[M * H];
  std::vector<float> tmp(N * H, 0.);
  for (size_t i = 0; i < score.size(); ++i) {
    score[i] = std::exp(score[i]);
  }
  for (size_t u = 0; u < csr.row_offsets.size() - 1; ++u) {
    for (mg_int eid = csr.row_offsets[u]; eid < csr.row_offsets[u+1]; ++eid) {
      mg_int v = csr.column_indices[eid];
      for (mg_int idx = 0; idx < H; ++idx) {
        tmp[v * H + idx] += score[eid * H + idx];
      }
    }
  }
  for (size_t eid = 0; eid < csr.column_indices.size(); ++eid) {
    for (mg_int i = 0; i < H; ++i) {
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
  const mg_int M = csr.column_indices.size();
  const int H = gdata->H;
  float* h_ret = new float[M * H];
  CUDA_CALL(cudaMemcpy(h_ret, gdata->ret, sizeof(float) * M * H, cudaMemcpyDeviceToHost));
  bool equal = utils::IterEqual(h_ret, truth->ret, M * H);
  std::cout << "Correct? " << equal << std::endl;
  delete [] h_ret;
}

}  // masked_mm

#endif
