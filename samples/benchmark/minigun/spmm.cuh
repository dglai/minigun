#ifndef SAMPLES_BENCHMARK_MINIGUN_SPMM_CUH_
#define SAMPLES_BENCHMARK_MINIGUN_SPMM_CUH_

#include <cuda_runtime.h>
#include "../../samples_io.h"
#include "../../samples_utils.h"

namespace spmm {

struct GData {
  int D = 0;  // feat size
  int H = 0;  // num heads
  float* ndata{nullptr};  // N*H*D
  float* weight{nullptr}; // M*H
  float* out{nullptr};    // N*H*D
};

struct SPMMFunctor {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    const int D = gdata->D;
    const int H = gdata->H;
    // each thread handles one attention head
    mg_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    mg_int stride_x = blockDim.x * gridDim.x;
    float* srcoff = gdata->ndata + (src * H * D);
    float* eidoff = gdata->weight + (eid * H);
    float* outoff = gdata->out + (dst * H * D);
    while (tx < D * H) {
      const int h = tx / D;
      atomicAdd(outoff + tx, __ldg(srcoff + tx) * __ldg(eidoff + h));
      tx += stride_x;
    }
  }
};

void InitGData(const utils::SampleCsr& csr, GData* gdata, GData* truth) {
  const mg_int N = csr.row_offsets.size() - 1;
  const mg_int M = csr.column_indices.size();
  const int H = gdata->H, D = gdata->D;
  std::vector<float> ndata(N * gdata->D * gdata->H), weight(M * gdata->H, 0.), out(N * gdata->D * gdata->H, 0.);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (float)rand() / RAND_MAX;
  }
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = (float)rand() / RAND_MAX;
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
  // compute truth
  truth->out = new float[N * H * D];
  std::fill(truth->out, truth->out + N * H * D, 0.);
  for (size_t u = 0; u < csr.row_offsets.size() - 1; u++) {
    for (mg_int eid = csr.row_offsets[u]; eid < csr.row_offsets[u + 1]; eid++) {
      mg_int v = csr.column_indices[eid];
      for (mg_int h = 0; h < H; h++) {
        for (mg_int idx = 0; idx < D; idx++)
          truth->out[((v * H) + h) * D + idx] +=
            ndata[((u * H) + h) * D + idx] * weight[eid * H + h];
      }
    }
  }
}

void FreeGData(GData* gdata, GData* truth) {
  CUDA_CALL(cudaFree(gdata->ndata));
  CUDA_CALL(cudaFree(gdata->weight));
  CUDA_CALL(cudaFree(gdata->out));
  delete [] truth->out;
}

void CheckResult(const utils::SampleCsr& csr, GData* gdata, GData* truth) {
  const mg_int N = csr.row_offsets.size() - 1;
  const mg_int M = csr.column_indices.size();
  const int H = gdata->H, D = gdata->D;
  float* h_out = new float[N * H * D];
  CUDA_CALL(cudaMemcpy(h_out, gdata->out, sizeof(float) * N * H * D, cudaMemcpyDeviceToHost));
  bool equal = utils::IterEqual(h_out, truth->out, N * H * D);
  std::cout << "Correct? " << equal << std::endl;
  delete [] h_out;
}

}  // masked_mm

#endif
