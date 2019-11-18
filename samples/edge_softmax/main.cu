/* Sample code for edge softmax.*/
#include <iostream>
#include <cstdlib>
#include <limits>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "../samples_utils.h"
#include "../samples_io.h"

struct GData {
  int32_t dim = 0;
  float* sum{nullptr};  // ndata
  float* max{nullptr};  // ndata
  float* score{nullptr};
  int* eid_mapping{nullptr};
};

/*
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
*/

// Max
struct EdgeMax {
  static __device__ __forceinline__ bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    int32_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride_x = blockDim.x * gridDim.x;
    const int32_t dim = gdata->dim;
    while (tx < dim) {
      gdata->max[dst * dim + tx] = max(
          gdata->max[dst * dim + dst * dim + tx],
          gdata->score[gdata->eid_mapping[eid] * dim + tx]);
      tx += stride_x;
    }
  }
};

// minus max, exp and sum
struct MinuxMaxExpSum {
  static __device__ __forceinline__ bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    int32_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride_x = blockDim.x * gridDim.x;
    const int32_t dim = gdata->dim;
    while (tx < dim) {
      gdata->score[eid * dim + tx] = expf(
          gdata->score[eid * dim + tx] - gdata->max[dst * dim + tx]);
      gdata->sum[dst * dim + tx] += gdata->score[eid * dim + tx];
      tx += stride_x;
    }
  }
};

// norm
struct Norm {
  static __device__ __forceinline__ bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    int32_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride_x = blockDim.x * gridDim.x;
    const int32_t dim = gdata->dim;
    while (tx < dim) {
      gdata->score[eid * dim + tx] /= gdata->sum[dst * dim + tx];
      tx += stride_x;
    }
  }
};

const int32_t D = 8;  // number of heads

std::vector<float> GroundTruth(
    const std::vector<int32_t>& row_offsets,
    const std::vector<int32_t>& column_indices,
    std::vector<float> score) {
  const size_t N = row_offsets.size() - 1;
  std::vector<float> tmp(N * D, 0.);
  for (size_t i = 0; i < score.size(); ++i) {
    score[i] = std::exp(score[i]);
  }
  for (size_t u = 0; u < row_offsets.size() - 1; ++u) {
    for (int32_t eid = row_offsets[u]; eid < row_offsets[u+1]; ++eid) {
      int32_t v = column_indices[eid];
      for (int32_t idx = 0; idx < D; ++idx) {
        tmp[v * D + idx] += score[eid * D + idx];
      }
    }
  }
  for (size_t eid = 0; eid < column_indices.size(); ++eid) {
    for (int32_t i = 0; i < D; ++i) {
      score[eid * D + i] /= tmp[column_indices[eid] * D + i];
    }
  }
  return score;
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
  int nt = utils::_FindNumThreads(D, 32);
  config.data_num_threads = nt;
  config.data_num_blocks = (M + nt - 1) / nt;
  CUDA_CALL(cudaStreamCreate(&config.stream));

  // Create feature data
  std::vector<float> vvec(N * D), evec(M * D);
  for (int32_t i = 0; i < N * D; ++i) {
    vvec[i] = std::numeric_limits<float>::lowest();
  }
  for (int32_t i = 0; i < M * D; ++i) {
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
  gdata.eid_mapping = csr_t_mapping.data;

  CUDA_CALL(cudaDeviceSynchronize());

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices, evec);
  //utils::VecPrint(truth);

  typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kDst> ConfigDst;
  minigun::advance::Advance<kDLGPU, int32_t, ConfigDst, GData, EdgeMax>(
      config, csr, csr_t, coo, &gdata, infront);
  typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kEdge> ConfigEdge;
  minigun::advance::Advance<kDLGPU, int32_t, ConfigEdge, GData, MinuxMaxExpSum>(
      config, csr, csr_t, coo, &gdata, infront);
  minigun::advance::Advance<kDLGPU, int32_t, ConfigEdge, GData, Norm>(
      config, csr, csr_t, coo, &gdata, infront);

  CUDA_CALL(cudaDeviceSynchronize());

  // verify output
  std::vector<float> rst(M * D);
  CUDA_CALL(cudaMemcpy(&rst[0], gdata.score, sizeof(float) * M * D, cudaMemcpyDeviceToHost));
  //utils::VecPrint(rst);

  std::cout << "Correct? " << utils::VecEqual(truth, rst) << std::endl;

  // free

  return 0;
}
