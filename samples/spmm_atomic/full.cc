/* Sample code for Sparse-Matrix-Dense Matrix multiplication.*/
#include <iostream>
#include <cstdlib>
#include <dmlc/omp.h>
#include <chrono>
#include <cassert>

#include <minigun/minigun.h>
#include "../samples_utils.h"
#include "../samples_io.h"

struct GData {
  int32_t dim = 0;
  float* cur{nullptr};
  float* next{nullptr};
  float* weight{nullptr};
  int* eid_mapping{nullptr};
};

struct SPMMFunctor {
  static inline void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    const int32_t D = gdata->dim;
    for (int32_t fid = 0; fid < D; ++fid) {
#pragma omp atomic
      gdata->next[dst * D + fid] += gdata->cur[src * D + fid] * gdata->weight[eid];
    }
  }
  static inline void ApplyEdgeReduce(
      int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float* val, GData* gdata) {}
  static inline int32_t GetFeatSize(GData* gdata) {
    return -1;
  }
  static inline float* GetOutBuf(GData* gdata) {
    return nullptr;
  }
  static inline int32_t GetOutOffset(int32_t idx, GData* gdata) {
    return idx;
  }
};

const int32_t D = 128;  // number of features

std::vector<float> GroundTruth(
    const std::vector<int32_t>& row_offsets,
    const std::vector<int32_t>& column_indices,
    const std::vector<float>& vdata,
    const std::vector<float>& edata) {
  std::vector<float> ret(vdata.size(), 0);
  for (size_t u = 0; u < row_offsets.size() - 1; ++u) {
    for (int32_t eid = row_offsets[u]; eid < row_offsets[u+1]; ++eid) {
      int32_t v = column_indices[eid];
      for (int32_t idx = 0; idx < D; ++idx) {
        ret[v * D + idx] += vdata[u * D + idx] * edata[eid];
      }
    }
  }
  return ret;
}

int main(int argc, char** argv) {
  assert(argc == 2);
  srand(42);

  // create graph
  std::vector<int32_t> row_offsets, column_indices;
  utils::CreateNPGraph(1000, 0.01, row_offsets, column_indices);
  const int32_t N = row_offsets.size() - 1;
  const int32_t M = column_indices.size();
  std::cout << "#nodes: " << N << " #edges: " << M
    << " #feats: " << D << std::endl;

  // copy graph to gpu
  minigun::IntCsr csr;
  csr.row_offsets.length = row_offsets.size();
  csr.row_offsets.data = &row_offsets[0];
  csr.column_indices.length = column_indices.size();
  csr.column_indices.data = &column_indices[0];
  csr.num_rows = N;
  csr.num_cols = N;

  // Create raw eid_mapping
  minigun::IntArray csr_mapping = utils::arange(0, M, kDLCPU);

  // Create csr_t and coo
  minigun::IntCsr csr_t;
  auto pack = utils::ToReverseCsr(csr, csr_mapping, kDLCPU);
  csr_t = pack.first;
  minigun::IntArray csr_t_mapping = pack.second;
  minigun::IntCoo coo;
  coo = utils::ToCoo(csr, kDLCPU);
  int mode = std::stoi(argv[1]);
  minigun::IntSpMat spmat = {nullptr, nullptr, nullptr};
  if (mode == 0)
    spmat = minigun::IntSpMat({&csr, nullptr, nullptr});
  else if (mode == 1)
    // NOTE(zihao): need a mapping to make it work
    spmat = minigun::IntSpMat({nullptr, &csr_t, nullptr});
  else if (mode == 2)
    spmat = minigun::IntSpMat({nullptr, nullptr, &coo});

  // Create Runtime Config, not used for cpu
  minigun::advance::RuntimeConfig config;
  config.ctx = {kDLCPU, 0};

  // Create feature data
  std::vector<float> vvec(N * D), evec(M);
  for (int32_t i = 0; i < N * D; ++i) {
    vvec[i] = (float)rand() / RAND_MAX;
  }
  for (int32_t i = 0; i < M; ++i) {
    evec[i] = (float)rand() / RAND_MAX;
  }

  // Copy feature data to gpu
  GData gdata;
  gdata.dim = D;
  std::vector<float> results(N * D);
  gdata.cur = &vvec[0];
  gdata.next = &results[0];
  gdata.weight = &evec[0];
  gdata.eid_mapping = csr_t_mapping.data;

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices,
      vvec, evec);

  typedef minigun::advance::Config<minigun::advance::kEdge> Config;
  minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, SPMMFunctor>(
      config, spmat, &gdata);

  // verify output
  std::cout << "Correct? " << utils::VecEqual(truth, results) << std::endl;

  const int K = 10;
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, SPMMFunctor>(
        config, spmat, &gdata);
  }

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, SPMMFunctor>(
        config, spmat, &gdata);
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Time(ms): " << elapsed_seconds.count() * 1e3 / K << std::endl;
  return 0;
}
