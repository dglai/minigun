/* Sample code for Sparse-Matrix-Vector multiplication.*/
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "../samples_utils.h"
#include "../samples_io.h"
#include "sys/time.h"

struct GData {
  float* cur{nullptr};
  float* next{nullptr};
  float* weight{nullptr};
  int* eid_mapping{nullptr};
};

struct SPMVFunctor {
  static __device__ __forceinline__ bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    atomicAdd(gdata->next + dst, gdata->cur[src] * gdata->weight[eid]);
  }
  static __device__ __forceinline__ void ApplyEdgeReduce(
    int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float& val, GData* gdata) {}
  static __device__ __forceinline__ int32_t GetFeatSize(GData* gdata) {
    return 1;
  }
  static __device__ __forceinline__ float* GetOutBuf(GData* gdata) {
    return gdata->next;
  }
};

std::vector<float> GroundTruth(
    const std::vector<int32_t>& row_offsets,
    const std::vector<int32_t>& column_indices,
    const std::vector<float>& vdata,
    const std::vector<float>& edata,
    const std::vector<int32_t>& infront_vec) {
  std::vector<float> ret(vdata.size(), 0);
  for (const int32_t u : infront_vec) {
    for (int32_t eid = row_offsets[u]; eid < row_offsets[u+1]; ++eid) {
      int32_t v = column_indices[eid];
      ret[v] += vdata[u] * edata[eid];
    }
  }
  return ret;
}

int main(int argc, char** argv) {
  srand(42);
  std::vector<int32_t> row_offsets, column_indices;
  utils::CreateNPGraph(10000, 0.01, row_offsets, column_indices);
  const int32_t N = row_offsets.size() - 1;
  const int32_t M = column_indices.size();
  std::cout << "#nodes: " << N << " #edges: " << M << std::endl;

  CUDA_CALL(cudaSetDevice(0));
  minigun::IntCsr csr;
  csr.row_offsets.length = row_offsets.size();
  CUDA_CALL(cudaMalloc(&csr.row_offsets.data, sizeof(int32_t) * row_offsets.size()));
  CUDA_CALL(cudaMemcpy(csr.row_offsets.data, &row_offsets[0],
        sizeof(int32_t) * row_offsets.size(), cudaMemcpyHostToDevice));
  csr.column_indices.length = column_indices.size();
  CUDA_CALL(cudaMalloc(&csr.column_indices.data, sizeof(int32_t) * column_indices.size()));
  CUDA_CALL(cudaMemcpy(csr.column_indices.data, &column_indices[0],
        sizeof(int32_t) * column_indices.size(), cudaMemcpyHostToDevice));

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

  // prepare frontiers
  minigun::IntArray infront, outfront;
  std::vector<int32_t> infront_vec;
  for (int32_t i = 3; i < 3 + 500; ++i) {
    infront_vec.push_back(i);
  }
  LOG(INFO) << "Input frontier size: " << infront_vec.size();
  infront.length = infront_vec.size();
  CUDA_CALL(cudaMalloc(&infront.data, sizeof(int32_t) * infront_vec.size()));
  CUDA_CALL(cudaMemcpy(infront.data, &infront_vec[0],
        sizeof(int32_t) * infront_vec.size(), cudaMemcpyHostToDevice));

  // Create stream
  minigun::advance::RuntimeConfig config;
  config.ctx = {kDLGPU, 0};
  config.data_num_blocks = 1;
  config.data_num_threads = 1;
  CUDA_CALL(cudaStreamCreate(&config.stream));

  // Create vdata, edata and copy to GPU
  std::vector<float> vvec(N), evec(M);
  for (int32_t i = 0; i < N; ++i) {
    vvec[i] = (float)rand() / RAND_MAX;
  }
  for (int32_t i = 0; i < M; ++i) {
    evec[i] = (float)rand() / RAND_MAX;
  }

  GData gdata;
  CUDA_CALL(cudaMalloc(&gdata.cur, sizeof(float) * N));
  CUDA_CALL(cudaMemcpy(gdata.cur, &vvec[0], sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&gdata.next, sizeof(float) * N));
  CUDA_CALL(cudaMemset(gdata.next, 0, sizeof(float) * N));
  CUDA_CALL(cudaMalloc(&gdata.weight, sizeof(float) * M));
  CUDA_CALL(cudaMemcpy(gdata.weight, &evec[0], sizeof(float) * M, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaDeviceSynchronize());

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices,
      vvec, evec, infront_vec);
  //utils::VecPrint(truth);

  typedef minigun::advance::Config<false, minigun::advance::kV2N, minigun::advance::kEdge> Config;
  minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, SPMVFunctor>(
      config, spmat, &gdata, infront, &outfront,
      utils::GPUAllocator::Get());

  CUDA_CALL(cudaDeviceSynchronize());

  // verify output
  std::vector<float> rst(N);
  CUDA_CALL(cudaMemcpy(&rst[0], gdata.next, sizeof(float) * N, cudaMemcpyDeviceToHost));
  //utils::VecPrint(rst);

  std::cout << "Correct? " << utils::VecEqual(truth, rst) << std::endl;

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, SPMVFunctor>(
        config, spmat, &gdata, infront, &outfront,
        utils::GPUAllocator::Get());
  }
  CUDA_CALL(cudaDeviceSynchronize());
  gettimeofday(&t1, nullptr);
  std::cout << "Time(ms): " << (double)(t1.tv_usec - t0.tv_usec) / K / 1000.0 << std::endl;

  // free

  return 0;
}
