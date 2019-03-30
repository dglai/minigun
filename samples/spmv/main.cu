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

struct GData {
  float* cur{nullptr};
  float* next{nullptr};
  float* weight{nullptr};
};

struct PRFunctor {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    atomicAdd(gdata->next + dst, gdata->cur[src] * gdata->weight[eid]);
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
      vvec, evec);
  //Print(truth);

  minigun::advance::Advance<GData, PRFunctor>(
      config, csr, d_gdata, infront, outfront);

  CUDA_CALL(cudaDeviceSynchronize());

  // verify output
  std::vector<float> rst(N);
  CUDA_CALL(cudaMemcpy(&rst[0], gdata.next, sizeof(float) * N, cudaMemcpyDeviceToHost));
  //Print(rst);

  std::cout << "Correct? " << Equal(truth, rst) << std::endl;

  // free

  return 0;
}
