/* Sample code for Sparse-Matrix-Vector multiplication.*/
#include <iostream>
#include <cstdlib>
#include <dmlc/omp.h>
#include <chrono>

#include <minigun/minigun.h>
#include "../samples_utils.h"

struct GData {
  float* cur{nullptr};
  float* next{nullptr};
  float* weight{nullptr};
};

struct SPMVFunctor {
  static inline bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
#pragma omp atomic
    gdata->next[dst] += gdata->cur[src] * gdata->weight[eid];
  }
};

std::vector<float> GroundTruth(
    const std::vector<mg_int>& row_offsets,
    const std::vector<mg_int>& column_indices,
    const std::vector<float>& vdata,
    const std::vector<float>& edata) {
  std::vector<float> ret(vdata.size(), 0);
  for (size_t u = 0; u < row_offsets.size() - 1; ++u) {
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

  minigun::Csr csr;
  minigun::IntArray1D infront;
  csr.row_offsets.length = row_offsets.size();
  csr.row_offsets.data = &row_offsets[0];
  csr.column_indices.length = column_indices.size();
  csr.column_indices.data = &column_indices[0];

  // Create Runtime Config, not used for cpu
  minigun::advance::RuntimeConfig config;
  config.ctx = {kDLCPU, 0};

  // Create vdata, edata and copy to GPU
  std::vector<float> vvec(N), evec(M);
  for (mg_int i = 0; i < N; ++i) {
    vvec[i] = (float)rand() / RAND_MAX;
  }
  for (mg_int i = 0; i < M; ++i) {
    evec[i] = (float)rand() / RAND_MAX;
  }

  GData gdata;
  std::vector<float> results(N);
  gdata.cur = &vvec[0];
  gdata.next = &results[0];
  gdata.weight = &evec[0];

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices,
      vvec, evec);
  //utils::VecPrint(truth);

  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLCPU, Config, GData, SPMVFunctor>(
      config, csr, &gdata, infront, nullptr,
      utils::CPUAllocator::Get());

  // verify output
  std::cout << "Correct? " << utils::VecEqual(truth, results) << std::endl;

  const int K = 10;
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, Config, GData, SPMVFunctor>(
        config, csr, &gdata, infront, nullptr,
        utils::CPUAllocator::Get());
  }

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, Config, GData, SPMVFunctor>(
        config, csr, &gdata, infront, nullptr,
        utils::CPUAllocator::Get());
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Time(ms): " << elapsed_seconds.count() * 1e3 / K << std::endl;
  return 0;
}
