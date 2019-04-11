/* Sample code for edge softmax.*/
#include <iostream>
#include <cstdlib>
#include <limits>
#include <time.h>
#include <omp.h>

#include <minigun/minigun.h>
#include "../samples_utils.h"

struct GData {
  mg_int dim = 0;
  float* sum{nullptr};  // ndata
  float* max{nullptr};  // ndata
  float* score{nullptr};
};

// Max
struct EdgeMax {
  static inline bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    const mg_int dim = gdata->dim;
#pragma omp parallel for
    for (mg_int fid = 0; fid < dim; ++fid) {
#pragma omp critical
      gdata->max[dst * dim + fid] = std::max(gdata->max[dst * dim + fid],
          gdata->score[eid * dim + fid]);
    }
  }
};

// minus max, exp and sum
struct MinuxMaxExpSum {
  static inline bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    const mg_int dim = gdata->dim;
#pragma omp parallel for
    for (mg_int fid = 0; fid < dim; ++fid) {
      gdata->score[eid * dim + fid] = expf(gdata->score[eid * dim + fid] - gdata->max[dst * dim + fid]);
#pragma omp atomic
      gdata->sum[dst * dim + fid] += gdata->score[eid * dim + fid];
    }
  }
};

// norm
struct Norm {
  static inline bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    const mg_int dim = gdata->dim;
    for (mg_int fid = 0; fid < dim; ++fid) {
      auto sum = gdata->sum[dst * dim + fid];
#pragma omp atomic
      gdata->score[eid * dim + fid] /= sum;
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
  minigun::Csr csr;
  csr.ctx.device_type = kDLCPU;
  minigun::IntArray1D infront, outfront;
  csr.row_offsets.length = row_offsets.size();
  csr.row_offsets.data = &row_offsets[0];
  csr.column_indices.length = column_indices.size();
  csr.column_indices.data = &column_indices[0];

  // Create Runtime Config, not used for cpu
  minigun::advance::RuntimeConfig config;

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
  std::vector<float> sum(N * D, 0);
  gdata.score = &evec[0];
  gdata.max = &vvec[0];
  gdata.sum = &sum[0];

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices, evec);
  //utils::VecPrint(truth);

  minigun::advance::Advance<GData, EdgeMax>(
      config, csr, &gdata, infront, outfront);
  minigun::advance::Advance<GData, MinuxMaxExpSum>(
      config, csr, &gdata, infront, outfront);
  minigun::advance::Advance<GData, Norm>(
      config, csr, &gdata, infront, outfront);

  // verify output
  std::cout << "Correct? " << utils::VecEqual(truth, evec) << std::endl;

  return 0;
}
