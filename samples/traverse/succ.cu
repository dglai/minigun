/* Sample code for Sparse-Matrix-Vector multiplication.*/
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "../samples_utils.h"

struct GData {
  float* cur{nullptr};
  float* next{nullptr};
  float* weight{nullptr};
};

struct SPMVFunctor {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    atomicAdd(gdata->next + dst, gdata->cur[src] * gdata->weight[eid]);
  }
};

std::vector<float> GroundTruth(
    const std::vector<mg_int>& row_offsets,
    const std::vector<mg_int>& column_indices,
    const std::vector<float>& vdata,
    const std::vector<float>& edata,
    const std::vector<mg_int>& infront_vec) {
  std::vector<float> ret(vdata.size(), 0);
  for (const mg_int u : infront_vec) {
    for (mg_int eid = row_offsets[u]; eid < row_offsets[u+1]; ++eid) {
      mg_int v = column_indices[eid];
      ret[v] += vdata[u] * edata[eid];
    }
  }
  return ret;
}

int main(int argc, char** argv) {
  srand(42);
  std::vector<mg_int> row_offsets, column_indices;
  utils::CreateNPGraph(1000, 0.01, row_offsets, column_indices);
  const mg_int N = row_offsets.size() - 1;
  const mg_int M = column_indices.size();
  std::cout << "#nodes: " << N << " #edges: " << M << std::endl;

  CUDA_CALL(cudaSetDevice(0));
  minigun::Csr csr;
  csr.row_offsets.length = row_offsets.size();
  CUDA_CALL(cudaMalloc(&csr.row_offsets.data, sizeof(mg_int) * row_offsets.size()));
  CUDA_CALL(cudaMemcpy(csr.row_offsets.data, &row_offsets[0],
        sizeof(mg_int) * row_offsets.size(), cudaMemcpyHostToDevice));
  csr.column_indices.length = column_indices.size();
  CUDA_CALL(cudaMalloc(&csr.column_indices.data, sizeof(mg_int) * column_indices.size()));
  CUDA_CALL(cudaMemcpy(csr.column_indices.data, &column_indices[0],
        sizeof(mg_int) * column_indices.size(), cudaMemcpyHostToDevice));

  // prepare frontiers
  minigun::IntArray1D infront, outfront;
  std::vector<mg_int> infront_vec;
  for (mg_int i = 3; i < 3 + 100; ++i) {
    infront_vec.push_back(i);
  }
  infront.length = infront_vec.size();
  CUDA_CALL(cudaMalloc(&infront.data, sizeof(mg_int) * infront_vec.size()));
  CUDA_CALL(cudaMemcpy(infront.data, &infront_vec[0],
        sizeof(mg_int) * infront_vec.size(), cudaMemcpyHostToDevice));

  // Create stream
  minigun::advance::RuntimeConfig config;
  config.data_num_blocks = 1;
  config.data_num_threads = 1;
  CUDA_CALL(cudaStreamCreate(&config.stream));

  // Create vdata, edata and copy to GPU
  std::vector<float> vvec(N), evec(M);
  for (mg_int i = 0; i < N; ++i) {
    vvec[i] = (float)rand() / RAND_MAX;
  }
  for (mg_int i = 0; i < M; ++i) {
    evec[i] = (float)rand() / RAND_MAX;
  }

  GData gdata;
  CUDA_CALL(cudaMalloc(&gdata.cur, sizeof(float) * N));
  CUDA_CALL(cudaMemcpy(gdata.cur, &vvec[0], sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&gdata.next, sizeof(float) * N));
  CUDA_CALL(cudaMemset(gdata.next, 0, sizeof(float) * N));
  CUDA_CALL(cudaMalloc(&gdata.weight, sizeof(float) * M));
  CUDA_CALL(cudaMemcpy(gdata.weight, &evec[0], sizeof(float) * M, cudaMemcpyHostToDevice));
  GData* d_gdata;
  CUDA_CALL(cudaMalloc(&d_gdata, sizeof(GData)));
  CUDA_CALL(cudaMemcpy(d_gdata, &gdata, sizeof(GData), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaDeviceSynchronize());

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices,
      vvec, evec, infront_vec);
  //utils::VecPrint(truth);

  typedef minigun::advance::Config<false, minigun::advance::kV2E> Config;
  minigun::advance::Advance<kDLGPU, Config, GData, SPMVFunctor>(
      config, csr, d_gdata, infront, outfront);

  CUDA_CALL(cudaDeviceSynchronize());

  // verify output
  std::vector<float> rst(N);
  CUDA_CALL(cudaMemcpy(&rst[0], gdata.next, sizeof(float) * N, cudaMemcpyDeviceToHost));
  //utils::VecPrint(rst);

  std::cout << "Correct? " << utils::VecEqual(truth, rst) << std::endl;

  // free

  return 0;
}
