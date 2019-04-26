/* Sample code for Dense-Dense => Sparse Matrix multiplication.*/
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "../samples_utils.h"

struct GData {
  mg_int dim = 0;
  float* ndata{nullptr};
  float* edata{nullptr};
};

struct MaskedMMFunctor {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    mg_int tx = threadIdx.x;
    const mg_int dim = gdata->dim;
    float sum = 0.;
    while (tx < dim) {
      sum += gdata->ndata[src * dim + tx] * gdata->ndata[dst * dim + tx];
      tx += blockDim.x;
    }
    atomicAdd(gdata->edata + eid, sum);
  }
};

const mg_int D = 128;  // number of features

std::vector<float> GroundTruth(
    const std::vector<mg_int>& row_offsets,
    const std::vector<mg_int>& column_indices,
    const std::vector<float>& vdata) {
  std::vector<float> ret(column_indices.size(), 0);
  for (size_t u = 0; u < row_offsets.size() - 1; ++u) {
    for (mg_int eid = row_offsets[u]; eid < row_offsets[u+1]; ++eid) {
      mg_int v = column_indices[eid];
      for (mg_int idx = 0; idx < D; ++idx) {
        ret[eid] += vdata[u * D + idx] * vdata[v * D + idx];
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
  minigun::IntArray1D infront;
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
  config.ctx = {kDLGPU, 0};
  config.data_num_threads = utils::_FindNumThreads(D, 32);
  config.data_num_blocks = 1;
  CUDA_CALL(cudaStreamCreate(&config.stream));

  // Create feature data
  std::vector<float> vvec(N * D), evec(M);
  for (mg_int i = 0; i < N * D; ++i) {
    vvec[i] = (float)rand() / RAND_MAX;
  }
  for (mg_int i = 0; i < M; ++i) {
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

  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, Config, GData, MaskedMMFunctor>(
      config, csr, &gdata, infront);

  CUDA_CALL(cudaDeviceSynchronize());

  // verify output
  std::vector<float> rst(M);
  CUDA_CALL(cudaMemcpy(&rst[0], gdata.edata, sizeof(float) * M, cudaMemcpyDeviceToHost));
  //utils::VecPrint(rst);

  std::cout << "Correct? " << utils::VecEqual(truth, rst) << std::endl;

  // free

  return 0;
}
