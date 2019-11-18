/* Sample code for Dense-Dense => Sparse Matrix multiplication.*/
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "../samples_utils.h"
#include "../samples_io.h"

struct GData {
  int32_t dim = 0;
  float* ndata{nullptr};
  float* edata{nullptr};
  int* eid_mapping{nullptr};
};

struct MaskedMMFunctor {
  static __device__ __forceinline__ bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    int32_t tx = threadIdx.x;
    const int32_t dim = gdata->dim;
    float sum = 0.;
    while (tx < dim) {
      sum += gdata->ndata[src * dim + tx] * gdata->ndata[dst * dim + tx];
      tx += blockDim.x;
    }
    atomicAdd(gdata->edata + eid, sum);
  }
};

const int32_t D = 128;  // number of features

std::vector<float> GroundTruth(
    const std::vector<int32_t>& row_offsets,
    const std::vector<int32_t>& column_indices,
    const std::vector<float>& vdata) {
  std::vector<float> ret(column_indices.size(), 0);
  for (size_t u = 0; u < row_offsets.size() - 1; ++u) {
    for (int32_t eid = row_offsets[u]; eid < row_offsets[u+1]; ++eid) {
      int32_t v = column_indices[eid];
      for (int32_t idx = 0; idx < D; ++idx) {
        ret[eid] += vdata[u * D + idx] * vdata[v * D + idx];
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

  // Create raw eid_mapping
  minigun::IntArray csr_mapping = utils::arange(0, M, kDLGPU);

  // Create csr_t and coo
  minigun::IntCsr csr_t;
  auto rev = utils::ToReverseCsr(csr, csr_mapping, kDLGPU);
  csr_t = rev.first;
  minigun::IntArray csr_t_mapping = rev.second;
  minigun::IntCoo coo;
  coo = utils::ToCoo(csr, kDLGPU);

  // Create stream
  minigun::advance::RuntimeConfig config;
  config.ctx = {kDLGPU, 0};
  config.data_num_threads = utils::_FindNumThreads(D, 32);
  config.data_num_blocks = 1;
  CUDA_CALL(cudaStreamCreate(&config.stream));

  // Create feature data
  std::vector<float> vvec(N * D), evec(M);
  for (int32_t i = 0; i < N * D; ++i) {
    vvec[i] = (float)rand() / RAND_MAX;
  }
  for (int32_t i = 0; i < M; ++i) {
    evec[i] = 0.;
  }

  // Copy feature data to gpu
  GData gdata;
  gdata.dim = D;
  CUDA_CALL(cudaMalloc(&gdata.ndata, sizeof(float) * N * D));
  CUDA_CALL(cudaMemcpy(gdata.ndata, &vvec[0], sizeof(float) * N * D, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&gdata.edata, sizeof(float) * M));
  CUDA_CALL(cudaMemcpy(gdata.edata, &evec[0], sizeof(float) * M, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaDeviceSynchronize());

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices, vvec);
  //utils::VecPrint(truth);

  typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kEdge> Config;
  minigun::advance::Advance<kDLGPU, int32_t, Config, GData, MaskedMMFunctor>(
      config, csr, csr_t, coo, &gdata, infront);

  CUDA_CALL(cudaDeviceSynchronize());

  // verify output
  std::vector<float> rst(M);
  CUDA_CALL(cudaMemcpy(&rst[0], gdata.edata, sizeof(float) * M, cudaMemcpyDeviceToHost));
  //utils::VecPrint(rst);

  std::cout << "Correct? " << utils::VecEqual(truth, rst) << std::endl;

  // free

  return 0;
}
