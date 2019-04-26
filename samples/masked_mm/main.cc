/* Sample code for Dense-Dense => Sparse Matrix multiplication.*/
#include <iostream>
#include <cstdlib>
#include <dmlc/omp.h>
#include <chrono>

#include <minigun/minigun.h>
#include "../samples_utils.h"

struct GData {
  mg_int dim = 0;
  float* ndata{nullptr};
  float* edata{nullptr};
};

struct MaskedMMFunctor {
  static inline bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    // only one block along the data dim
    const mg_int dim = gdata->dim;
    float sum = 0.;
    for (mg_int fid = 0; fid < dim; ++fid) {
      sum += gdata->ndata[src * dim + fid] * gdata->ndata[dst * dim + fid];
    }
#pragma omp atomic
    gdata->edata[eid] += sum;
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
  minigun::Csr csr;
  minigun::IntArray1D infront;
  csr.row_offsets.length = row_offsets.size();
  csr.row_offsets.data = &row_offsets[0];
  csr.column_indices.length = column_indices.size();
  csr.column_indices.data = &column_indices[0];

  // Create Runtime Config, not used for cpu
  minigun::advance::RuntimeConfig config;
  config.ctx = {kDLCPU, 0};

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
  gdata.ndata = &vvec[0];
  gdata.edata = &evec[0];

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices, vvec);
  //utils::VecPrint(truth);

  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLCPU, Config, GData, MaskedMMFunctor>(
      config, csr, &gdata, infront);

  // verify output
  std::cout << "Correct? " << utils::VecEqual(truth, evec) << std::endl;

  const int K = 10;
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, Config, GData, MaskedMMFunctor>(
        config, csr, &gdata, infront);
  }

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, Config, GData, MaskedMMFunctor>(
        config, csr, &gdata, infront);
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Time(ms): " << elapsed_seconds.count() * 1e3 / K << std::endl;
  return 0;
  return 0;
}
