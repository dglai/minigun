#include <minigun/minigun.h>

#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

template<typename T>
void Print(const std::vector<T>& vec) {
  std::cout << "[";
  for (T t : vec) {
    std::cout << t << " ";
  }
  std::cout << "]" << std::endl;
}

template<typename T>
bool Equal(const std::vector<T>& v1,
           const std::vector<T>& v2) {
  if (v1.size() != v2.size()) return false;
  for (size_t i = 0; i < v1.size(); ++i) {
    if (fabs(v1[i] - v2[i]) >= 1e-5) {
      return false;
    }
  }
  return true;
}

struct VData {
  float* cur{nullptr};
  float* next{nullptr};
};

struct PRFunctor {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, VData vdata, float* edata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, VData vdata, float* edata) {
    atomicAdd(vdata.next + dst, vdata.cur[src] * edata[eid]);
  }
};

void CreateGraph(int64_t N, float P, std::vector<mg_int>& row_offsets,
    std::vector<mg_int>& column_indices) {
  row_offsets.resize(N+1, 0);
  row_offsets[0] = 0;
  for (mg_int u = 0; u < N; ++u) {
    for (mg_int v = 0; v < N; ++v) {
      if ((float)rand() / RAND_MAX < P) {
        column_indices.push_back(v);
      }
    }
    row_offsets[u + 1] = column_indices.size();
  }
}

std::vector<float> GroundTruth(
    const std::vector<mg_int>& row_offsets,
    const std::vector<mg_int>& column_indices,
    const std::vector<float>& vdata,
    const std::vector<float>& edata) {
  std::vector<float> ret(vdata.size(), 0);
  for (mg_int u = 0; u < row_offsets.size() - 1; ++u) {
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
  CreateGraph(1000, 0.01, row_offsets, column_indices);
  const mg_int N = row_offsets.size() - 1;
  const mg_int M = column_indices.size();
  std::cout << "#nodes: " << N << " #edges: " << M << std::endl;

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
  CUDA_CALL(cudaStreamCreate(&config.stream));

  // Create vdata, edata and copy to GPU
  std::vector<float> vvec(N), evec(M);
  for (mg_int i = 0; i < N; ++i) {
    vvec[i] = (float)rand() / RAND_MAX;
  }
  for (mg_int i = 0; i < M; ++i) {
    evec[i] = (float)rand() / RAND_MAX;
  }

  VData vdata;
  CUDA_CALL(cudaMalloc(&vdata.cur, sizeof(float) * N));
  CUDA_CALL(cudaMemcpy(vdata.cur, &vvec[0], sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&vdata.next, sizeof(float) * N));
  CUDA_CALL(cudaMemset(vdata.next, 0, sizeof(float) * N));
  float *edata = nullptr;
  CUDA_CALL(cudaMalloc(&edata, sizeof(float) * M));
  CUDA_CALL(cudaMemcpy(edata, &evec[0], sizeof(float) * M, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaDeviceSynchronize());

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices,
      vvec, evec);
  //Print(truth);

  minigun::advance::Advance<VData, float*, PRFunctor>(
      config, csr, vdata, edata, infront, outfront);

  CUDA_CALL(cudaDeviceSynchronize());

  // verify output
  std::vector<float> rst(N);
  CUDA_CALL(cudaMemcpy(&rst[0], vdata.next, sizeof(float) * N, cudaMemcpyDeviceToHost));
  //Print(rst);

  std::cout << "Correct? " << Equal(truth, rst) << std::endl;

  // free

  return 0;
}
