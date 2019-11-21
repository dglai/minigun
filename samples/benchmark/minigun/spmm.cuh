#ifndef SAMPLES_BENCHMARK_MINIGUN_SPMM_CUH_
#define SAMPLES_BENCHMARK_MINIGUN_SPMM_CUH_

#include <cuda_runtime.h>
#include <cassert>
#include "../../samples_io.h"
#include "../../samples_utils.h"

namespace spmm {

struct GData {
  int D = 0;  // feat size
  float* ndata{nullptr};  // N*D
  float* weight{nullptr}; // M
  float* out{nullptr};    // N*D
  int* eid_mapping{nullptr};
  __host__ __device__ __forceinline__ int GetFeatSize() {
    return D;
  }
  template <typename Functor>
  __host__ __device__ __forceinline__ float* GetOutBuf() {
    return out;
  }
};

struct SPMMFunctor {
  static __device__ __forceinline__ bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {}
  static __device__ __forceinline__ void ApplyEdgeReduce(
      int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float& val, GData* gdata) {
    const int D = gdata->D;
    // each thread handles one attention head
    float* srcoff = gdata->ndata + src * D;
    float* eidoff = gdata->weight + gdata->eid_mapping[eid];
    val += __ldg(srcoff + feat_idx) * __ldg(eidoff);
  }
};

void InitGData(const utils::SampleCsr& csr, const::minigun::IntArray eid_mapping,
    GData* gdata, GData* truth) {
  const int32_t N = csr.row_offsets.size() - 1;
  const int32_t M = csr.column_indices.size();
  const int D = gdata->D;
  std::vector<float> ndata(N * gdata->D), weight(M, 0.), out(N * gdata->D, 0.);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (float)rand() / RAND_MAX;
  }
  for (size_t i = 0; i < weight.size(); ++i) {
    // XXX: weight has to be the same across edges because transpose function did not change weights
    weight[i] = 3.45;
  }
  CUDA_CALL(cudaMalloc(&(gdata->ndata), sizeof(float) * ndata.size()));
  CUDA_CALL(cudaMemcpy(gdata->ndata, &ndata[0],
        sizeof(float) * ndata.size(), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->out), sizeof(float) * out.size()));
  CUDA_CALL(cudaMemcpy(gdata->out, &out[0],
        sizeof(float) * out.size(), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->weight), sizeof(float) * weight.size()));
  CUDA_CALL(cudaMemcpy(gdata->weight, &weight[0],
        sizeof(float) * weight.size(), cudaMemcpyHostToDevice));
  gdata->eid_mapping = eid_mapping.data;
  // compute truth
  truth->out = new float[N * D];
  std::fill(truth->out, truth->out + N * D, 0.);
  for (size_t u = 0; u < csr.row_offsets.size() - 1; u++) {
    for (int32_t eid = csr.row_offsets[u]; eid < csr.row_offsets[u + 1]; eid++) {
      int32_t v = csr.column_indices[eid];
      for (int32_t idx = 0; idx < D; idx++) {
        truth->out[v * D + idx] += ndata[u * D + idx] * weight[eid];
      }
    }
  }
}

void ResetGData(GData* gdata, size_t N) {
  cudaMemset(gdata->out, 0, gdata->D * N *sizeof(float));
}

void FreeGData(GData* gdata, GData* truth) {
  CUDA_CALL(cudaFree(gdata->ndata));
  CUDA_CALL(cudaFree(gdata->weight));
  CUDA_CALL(cudaFree(gdata->out));
  delete [] truth->out;
}

void CheckResult(const utils::SampleCsr& csr, GData* gdata, GData* truth) {
  const int32_t N = csr.row_offsets.size() - 1;
  const int32_t M = csr.column_indices.size();
  const int D = gdata->D;
  float* h_out = new float[N * D];
  CUDA_CALL(cudaMemcpy(h_out, gdata->out, sizeof(float) * N * D, cudaMemcpyDeviceToHost));
  bool equal = utils::IterEqual(h_out, truth->out, N * D);
  /*
  if (!equal) {
    for (int i = 0; i < N; ++i) {
      std::cout << h_out[i] << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < N; ++i) {
      std::cout << truth->out[i] << " ";
    }
    std::cout << "\n";
  }
  */
  assert(equal);
  //std::cout << "Correct? " << equal << std::endl;
  delete [] h_out;
}

}  // masked_mm

#endif
