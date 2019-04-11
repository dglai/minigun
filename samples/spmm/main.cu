/* Sample code for Sparse-Matrix-Dense Matrix multiplication.*/
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "../samples_utils.h"

struct GData {
  mg_int dim = 0;
  float* cur{nullptr};
  float* next{nullptr};
  float* weight{nullptr};
};

struct SPMMFunctor {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    mg_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    mg_int stride_x = blockDim.x * gridDim.x;
    while (tx < gdata->dim) {
      atomicAdd(gdata->next + dst * gdata->dim + tx,
                gdata->cur[src * gdata->dim + tx] * gdata->weight[eid]);
      tx += stride_x;
    }
  }
};

const mg_int D = 128;  // number of features

std::vector<float> GroundTruth(
    const std::vector<mg_int>& row_offsets,
    const std::vector<mg_int>& column_indices,
    const std::vector<float>& vdata,
    const std::vector<float>& edata) {
  std::vector<float> ret(vdata.size(), 0);
  for (size_t u = 0; u < row_offsets.size() - 1; ++u) {
    for (mg_int eid = row_offsets[u]; eid < row_offsets[u+1]; ++eid) {
      mg_int v = column_indices[eid];
      for (mg_int idx = 0; idx < D; ++idx) {
        ret[v * D + idx] += vdata[u * D + idx] * edata[eid];
      }
    }
  }
  return ret;
}

int main(int argc, char** argv) {
  srand(42);

  // create graph
  std::vector<mg_int> row_offsets, column_indices;
  utils::CreateNPGraph(1000, 0.01, row_offsets, column_indices);
  const mg_int N = row_offsets.size() - 1;
  const mg_int M = column_indices.size();
  std::cout << "#nodes: " << N << " #edges: " << M
    << " #feats: " << D << std::endl;

  // copy graph to gpu
  CUDA_CALL(cudaSetDevice(0));
  minigun::Csr csr;
  minigun::IntArray1D infront, outfront;
  csr.row_offsets.length = row_offsets.size();
  CUDA_CALL(cudaMalloc(&csr.row_offsets.data, sizeof(mg_int) * row_offsets.size()));
  CUDA_CALL(cudaMemcpy(csr.row_offsets.data, &row_offsets[0],
        sizeof(mg_int) * row_offsets.size(), cudaMemcpyHostToDevice));
  csr.column_indices.length = column_indices.size();
  CUDA_CALL(cudaMalloc(&csr.column_indices.data, sizeof(mg_int) * column_indices.size()));
  CUDA_CALL(cudaMemcpy(csr.column_indices.data, &column_indices[0],
        sizeof(mg_int) * column_indices.size(), cudaMemcpyHostToDevice));

  // Create stream
  minigun::advance::RuntimeConfig config;
  int nt = utils::_FindNumThreads(D, 32);
  config.data_num_threads = nt;
  config.data_num_blocks = (M + nt - 1) / nt;
  CUDA_CALL(cudaStreamCreate(&config.stream));

  // Create feature data
  std::vector<float> vvec(N * D), evec(M);
  for (mg_int i = 0; i < N * D; ++i) {
    vvec[i] = (float)rand() / RAND_MAX;
  }
  for (mg_int i = 0; i < M; ++i) {
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
  GData* d_gdata;
  CUDA_CALL(cudaMalloc(&d_gdata, sizeof(GData)));
  CUDA_CALL(cudaMemcpy(d_gdata, &gdata, sizeof(GData), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaDeviceSynchronize());

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices,
      vvec, evec);

  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, Config, GData, SPMMFunctor>(
      config, csr, d_gdata, infront, outfront);

  CUDA_CALL(cudaDeviceSynchronize());

  // verify output
  std::vector<float> rst(N * D);
  CUDA_CALL(cudaMemcpy(&rst[0], gdata.next, sizeof(float) * N * D, cudaMemcpyDeviceToHost));

  std::cout << "Correct? " << utils::VecEqual(truth, rst) << std::endl;

  // free

  return 0;
}
