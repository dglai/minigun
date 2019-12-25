/* Sample code for Sparse-Matrix-Dense Matrix multiplication.*/
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "../samples_utils.h"
#include "../samples_io.h"

struct GData {
  int32_t dim = 0;
  float* cur{nullptr};
  float* next{nullptr};
  float* weight{nullptr};
  int* eid_mapping{nullptr};
};

struct SPMMFunctor {
  static __device__ __forceinline__ bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {}
  static __device__ __forceinline__ void ApplyEdgeReduce(
      int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float* val, GData* gdata) {
    *val += gdata->cur[src * gdata->dim + feat_idx] * gdata->weight[gdata->eid_mapping[eid]];
  }
  static __device__ __forceinline__ int32_t GetFeatSize(GData* gdata) {
    return gdata->dim;
  }
  static __device__ __forceinline__ float* GetOutBuf(GData* gdata) {
    return gdata->next;
  }
  static __device__ __forceinline__ int32_t GetOutOffset(int32_t idx, GData* gdata) {
    return idx;
  }
};

const int32_t D = 128;  // number of features

std::vector<float> GroundTruth(
    const std::vector<int32_t>& row_offsets,
    const std::vector<int32_t>& column_indices,
    const std::vector<float>& vdata,
    const std::vector<float>& edata) {
  std::vector<float> ret(vdata.size(), 0);
  for (size_t u = 0; u < row_offsets.size() - 1; ++u) {
    for (int32_t eid = row_offsets[u]; eid < row_offsets[u+1]; ++eid) {
      int32_t v = column_indices[eid];
      for (int32_t idx = 0; idx < D; ++idx) {
        ret[v * D + idx] += vdata[u * D + idx] * edata[eid];
      }
    }
  }
  return ret;
}

int main(int argc, char** argv) {
  srand(42);

  // create graph
  std::vector<int32_t> row_offsets, column_indices;
  utils::CreateNPGraph(1000, 0.01, row_offsets, column_indices);
  const int32_t N = row_offsets.size() - 1;
  const int32_t M = column_indices.size();
  std::cout << "#nodes: " << N << " #edges: " << M
    << " #feats: " << D << std::endl;

  // copy graph to gpu
  CUDA_CALL(cudaSetDevice(0));
  minigun::IntCsr csr;
  minigun::IntArray infront;
  csr.row_offsets.length = row_offsets.size();
  CUDA_CALL(cudaMalloc(&csr.row_offsets.data, sizeof(int32_t) * row_offsets.size()));
  CUDA_CALL(cudaMemcpy(csr.row_offsets.data, &row_offsets[0],
        sizeof(int32_t) * row_offsets.size(), cudaMemcpyHostToDevice));
  csr.column_indices.length = column_indices.size();
  CUDA_CALL(cudaMalloc(&csr.column_indices.data, sizeof(int32_t) * column_indices.size()));
  CUDA_CALL(cudaMemcpy(csr.column_indices.data, &column_indices[0],
        sizeof(int32_t) * column_indices.size(), cudaMemcpyHostToDevice));
  csr.num_rows = N;
  csr.num_cols = N;

  // Create raw eid_mapping
  minigun::IntArray csr_mapping = utils::arange(0, M, kDLGPU);

  // Create csr_t and coo
  minigun::IntCsr csr_t;
  auto pack = utils::ToReverseCsr(csr, csr_mapping, kDLGPU);
  csr_t = pack.first;
  minigun::IntArray csr_t_mapping = pack.second;
  minigun::IntCoo coo;
  coo = utils::ToCoo(csr, kDLGPU);
  minigun::IntSpMat spmat = {&csr, &csr_t, &coo};

  // Create stream
  minigun::advance::RuntimeConfig config;
  config.ctx = {kDLGPU, 0};
  int nt = 1; //utils::_FindNumThreads(D, 32);
  config.data_num_threads = nt;
  config.data_num_blocks = 1;
  CUDA_CALL(cudaStreamCreate(&config.stream));

  // Create feature data
  std::vector<float> vvec(N * D), evec(M);
  for (int32_t i = 0; i < N * D; ++i) {
    vvec[i] = (float)rand() / RAND_MAX;
  }
  for (int32_t i = 0; i < M; ++i) {
    evec[i] = (float)rand() / RAND_MAX;
  }

  // Copy feature data to gpu
  GData gdata;
  gdata.dim = D;
  CUDA_CALL(cudaMalloc(&gdata.cur, sizeof(float) * N * D));
  CUDA_CALL(cudaMemcpy(gdata.cur, &vvec[0], sizeof(float) * N * D, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&gdata.next, sizeof(float) * N * D));
  CUDA_CALL(cudaMemset(gdata.next, 0, sizeof(float) * N * D));
  CUDA_CALL(cudaMalloc(&gdata.weight, sizeof(float) * M));
  CUDA_CALL(cudaMemcpy(gdata.weight, &evec[0], sizeof(float) * M, cudaMemcpyHostToDevice));
  gdata.eid_mapping = csr_t_mapping.data;

  CUDA_CALL(cudaDeviceSynchronize());

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices,
      vvec, evec);

  typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kDst> Config;
  minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, SPMMFunctor>(
      config, spmat, &gdata, infront, nullptr);

  CUDA_CALL(cudaDeviceSynchronize());

  // verify output
  std::vector<float> rst(N * D);
  CUDA_CALL(cudaMemcpy(&rst[0], gdata.next, sizeof(float) * N * D, cudaMemcpyDeviceToHost));

  std::cout << "Correct? " << utils::VecEqual(truth, rst) << std::endl;

  // free

  return 0;
}
