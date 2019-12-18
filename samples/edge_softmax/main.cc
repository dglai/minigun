/* Sample code for edge softmax.*/
#include <iostream>
#include <cstdlib>
#include <limits>
#include <dmlc/omp.h>
#include <chrono>

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

// Max
struct EdgeMax {
  static inline bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {}
  static inline void ApplyEdgeReduce(
      int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float& val, GData* gdata) {
    const int32_t dim = gdata->dim;
    val = std::max(val, gdata->score[gdata->eid_mapping[eid] * dim + feat_idx]);
  }
  static inline int32_t GetFeatSize(GData* gdata) {
    return gdata->dim;
  }
  static inline float* GetOutBuf(GData* gdata) {
    return gdata->max;
  }
  static inline int32_t GetOutOffset(int32_t idx, GData* gdata) {
    return idx;
  }
};

// minus max, exp and sum
struct MinuxMaxExpSum {
  static inline bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {}
  static inline void ApplyEdgeReduce(
      int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float& val, GData* gdata) {
    const int32_t dim = gdata->dim;
    gdata->score[gdata->eid_mapping[eid] * dim + feat_idx] =
        expf(gdata->score[gdata->eid_mapping[eid] * dim + feat_idx] - gdata->max[dst * dim + feat_idx]);
    val += gdata->score[gdata->eid_mapping[eid] * dim + feat_idx];
  }
  static inline int32_t GetFeatSize(GData* gdata) {
    return gdata->dim;
  }
  static inline float* GetOutBuf(GData* gdata) {
    return gdata->sum;
  }
  static inline int32_t GetOutOffset(int32_t idx, GData* gdata) {
    return idx;
  }
};

// norm
struct Norm {
  static inline bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    const int32_t dim = gdata->dim;
    for (int32_t fid = 0; fid < dim; ++fid) {
      gdata->score[eid * dim + fid] /= gdata->sum[dst * dim + fid];
    }
  }
  static inline void ApplyEdgeReduce(
      int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float& val, GData* gdata) {}
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
  minigun::IntCsr csr;
  minigun::IntArray infront;
  csr.row_offsets.length = row_offsets.size();
  csr.row_offsets.data = &row_offsets[0];
  csr.column_indices.length = column_indices.size();
  csr.column_indices.data = &column_indices[0];

  // Create raw eid_mapping
  minigun::IntArray csr_mapping = utils::arange(0, M, kDLCPU);

  // Create csr_t and coo
  minigun::IntCsr csr_t;
  auto pack = utils::ToReverseCsr(csr, csr_mapping, kDLCPU);
  csr_t = pack.first;
  minigun::IntArray csr_t_mapping = pack.second;
  minigun::IntCoo coo;
  coo = utils::ToCoo(csr, kDLCPU);
  minigun::IntSpMat spmat = {&csr, &csr_t, &coo};

  // Create Runtime Config, not used for cpu
  minigun::advance::RuntimeConfig config;
  config.ctx = {kDLCPU, 0};

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
  std::vector<float> sum(N * D, 0);
  gdata.score = &evec[0];
  gdata.max = &vvec[0];
  gdata.sum = &sum[0];
  gdata.eid_mapping = csr_t_mapping.data;

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices, evec);
  //utils::VecPrint(truth);

  typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kDst> ConfigDst;
  typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kEdge> ConfigEdge;
  minigun::advance::Advance<kDLCPU, int32_t, float, ConfigDst, GData, EdgeMax>(
      config, spmat, &gdata, infront);
  minigun::advance::Advance<kDLCPU, int32_t, float, ConfigDst, GData, MinuxMaxExpSum>(
      config, spmat, &gdata, infront);
  minigun::advance::Advance<kDLCPU, int32_t, float, ConfigEdge, GData, Norm>(
      config, spmat, &gdata, infront);

  // verify output
  std::cout << "Correct? " << utils::VecEqual(truth, evec) << std::endl;

  const int K = 10;
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, int32_t, float, ConfigDst, GData, EdgeMax>(
        config, spmat, &gdata, infront);
    minigun::advance::Advance<kDLCPU, int32_t, float, ConfigDst, GData, MinuxMaxExpSum>(
        config, spmat, &gdata, infront);
    minigun::advance::Advance<kDLCPU, int32_t, float, ConfigEdge, GData, Norm>(
        config, spmat, &gdata, infront);
  }

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, int32_t, float, ConfigDst, GData, EdgeMax>(
        config, spmat, &gdata, infront);
    minigun::advance::Advance<kDLCPU, int32_t, float, ConfigDst, GData, MinuxMaxExpSum>(
        config, spmat, &gdata, infront);
    minigun::advance::Advance<kDLCPU, int32_t, float, ConfigEdge, GData, Norm>(
        config, spmat, &gdata, infront);
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Time(ms): " << elapsed_seconds.count() * 1e3 / K << std::endl;
  return 0;
}
