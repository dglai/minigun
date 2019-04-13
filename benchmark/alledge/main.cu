/* Benchmark code for all edge policy. */
#include <iostream>
#include <cstdlib>
#include <limits>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "../benchmark_utils.h"

struct GData {
  mg_int dim = 0;
  float* ndata{nullptr}; // ndata
  float* sum{nullptr};
  float* max{nullptr};
  float* score{nullptr}; // edata
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

// masked mm
struct MaskedMM {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    // only one block along the data dim
    mg_int tx = threadIdx.x;
    const mg_int dim = gdata->dim;
    float sum = 0.;
    while (tx < dim) {
      sum += gdata->ndata[src * dim + tx] * gdata->ndata[dst * dim + tx];
      tx += blockDim.x;
    }
    atomicAdd(gdata->score + eid, sum);
  }
};

const mg_int D = 512; // hidden dimension.

std::vector<float> GroundTruth(
    const std::vector<mg_int>& row_offsets,
    const std::vector<mg_int>& column_indices,
    const std::vector<float>& vdata) {
  std::vector<float> ret(column_indices.size(), 0);
  for (size_t u = 0; u < row_offsets.size() - 1; u++) {
    for (mg_int eid = row_offsets[u]; eid < row_offsets[u + 1]; eid++) {
      mg_int v = column_indices[eid];
      for (mg_int idx = 0; idx < D; idx++)
        ret[eid] += vdata[u * D + idx] * vdata[v * D + idx];
    }
  }
  return ret;
}

int main(int argc, char **argv) {
  srand(42);

  // create graph
  std::vector<mg_int> row_offsets, column_indices;
  utils::CreateSparseBatch1(1, row_offsets, column_indices);
  const mg_int N = row_offsets.size() - 1;
  const mg_int M = column_indices.size();
  std::cout << "#nodes: " << N << " #edges: " << M << " #feats: " << D << std::endl; 
  
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
    evec[i] = 0.;
  }

  // Copy feature data to gpu
  GData gdata;
  gdata.dim = D;
  CUDA_CALL(cudaMalloc(&gdata.ndata, sizeof(float) * N * D));
  CUDA_CALL(cudaMemcpy(gdata.ndata, &vvec[0], sizeof(float) * N * D, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&gdata.score, sizeof(float) * M));
  CUDA_CALL(cudaMemcpy(gdata.score, &evec[0], sizeof(float) * M, cudaMemcpyHostToDevice));
  GData* d_gdata;
  CUDA_CALL(cudaMalloc(&d_gdata, sizeof(GData)));
  CUDA_CALL(cudaMemcpy(d_gdata, &gdata, sizeof(GData), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaDeviceSynchronize());

  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, Config, GData, MaskedMM>(
      config, csr, d_gdata, infront, outfront);

  CUDA_CALL(cudaDeviceSynchronize());

  /*
  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, Config, GData, EdgeMax>(
      config, csr, d_gdata, infront, outfront);
  minigun::advance::Advance<kDLGPU, Config, GData, MinuxMaxExpSum>(
      config, csr, d_gdata, infront, outfront);
  minigun::advance::Advance<kDLGPU, Config, GData, Norm>(
      config, csr, d_gdata, infront, outfront);

  CUDA_CALL(cudaDeviceSynchronize());
  */

  // output
  std::vector<float> rst(M);
  CUDA_CALL(cudaMemcpy(&rst[0], gdata.score, sizeof(float) * M, cudaMemcpyDeviceToHost));
  utils::VecPrint(rst);

  return 0;
}