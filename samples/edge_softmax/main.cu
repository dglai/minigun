/* Sample code for edge softmax.*/
#include <iostream>
#include <cstdlib>
#include <limits>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "../samples_utils.h"

struct GData {
  mg_int dim = 0;
  float* sum{nullptr};  // ndata
  float* max{nullptr};  // ndata
  float* score{nullptr};
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
    mg_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    mg_int stride_x = blockDim.x * gridDim.x;
    const mg_int dim = gdata->dim;
    while (tx < dim) {
      MyAtomicMax(gdata->max + dst * dim + tx, gdata->score[eid * dim + tx]);
      tx += stride_x;
    }
  }
};

// minus max, exp and sum
struct MinuxMaxExpSum {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    mg_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    mg_int stride_x = blockDim.x * gridDim.x;
    const mg_int dim = gdata->dim;
    while (tx < dim) {
      gdata->score[eid * dim + tx] = expf(
          gdata->score[eid * dim + tx] - gdata->max[dst * dim + tx]);
      atomicAdd(gdata->sum + dst * dim + tx, gdata->score[eid * dim + tx]);
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
    mg_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    mg_int stride_x = blockDim.x * gridDim.x;
    const mg_int dim = gdata->dim;
    while (tx < dim) {
      gdata->score[eid * dim + tx] /= gdata->sum[dst * dim + tx];
      tx += stride_x;
    }
  }
};

const mg_int D = 8;  // number of heads

std::vector<float> GroundTruth(
    const std::vector<mg_int>& row_offsets,
    const std::vector<mg_int>& column_indices,
    std::vector<float> score) {
  const size_t N = row_offsets.size() - 1;
  std::vector<float> tmp(N * D, 0.);
  for (size_t i = 0; i < score.size(); ++i) {
    score[i] = std::exp(score[i]);
  }
  for (size_t u = 0; u < row_offsets.size() - 1; ++u) {
    for (mg_int eid = row_offsets[u]; eid < row_offsets[u+1]; ++eid) {
      mg_int v = column_indices[eid];
      for (mg_int idx = 0; idx < D; ++idx) {
        tmp[v * D + idx] += score[eid * D + idx];
      }
    }
  }
  for (size_t eid = 0; eid < column_indices.size(); ++eid) {
    for (mg_int i = 0; i < D; ++i) {
      score[eid * D + i] /= tmp[column_indices[eid] * D + i];
    }
  }
  return score;
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
  int nt = utils::_FindNumThreads(D, 32);
  config.data_num_threads = nt;
  config.data_num_blocks = (M + nt - 1) / nt;
  CUDA_CALL(cudaStreamCreate(&config.stream));

  // Create feature data
  std::vector<float> vvec(N * D), evec(M * D);
  for (mg_int i = 0; i < N * D; ++i) {
    vvec[i] = std::numeric_limits<float>::lowest();
  }
  for (mg_int i = 0; i < M * D; ++i) {
    evec[i] = (float)rand() / RAND_MAX - 0.5;
  }
  //utils::VecPrint(evec);

  // Copy feature data to gpu
  GData gdata;
  gdata.dim = D;
  CUDA_CALL(cudaMalloc(&gdata.sum, sizeof(float) * N * D));
  CUDA_CALL(cudaMemset(gdata.sum, 0, sizeof(float) * N * D));
  CUDA_CALL(cudaMalloc(&gdata.max, sizeof(float) * N * D));
  CUDA_CALL(cudaMemcpy(gdata.max, &vvec[0], sizeof(float) * N * D, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&gdata.score, sizeof(float) * M * D));
  CUDA_CALL(cudaMemcpy(gdata.score, &evec[0], sizeof(float) * M * D, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaDeviceSynchronize());

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices, evec);
  //utils::VecPrint(truth);

  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, Config, GData, EdgeMax>(
      config, csr, &gdata, infront);
  minigun::advance::Advance<kDLGPU, Config, GData, MinuxMaxExpSum>(
      config, csr, &gdata, infront);
  minigun::advance::Advance<kDLGPU, Config, GData, Norm>(
      config, csr, &gdata, infront);

  CUDA_CALL(cudaDeviceSynchronize());

  // verify output
  std::vector<float> rst(M * D);
  CUDA_CALL(cudaMemcpy(&rst[0], gdata.score, sizeof(float) * M * D, cudaMemcpyDeviceToHost));
  //utils::VecPrint(rst);

  std::cout << "Correct? " << utils::VecEqual(truth, rst) << std::endl;

  // free

  return 0;
}
