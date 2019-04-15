#ifndef __GROUND_TRUTH_OPS_H
#define __GROUND_TRUTH_OPS_H

#include <iostream>

namespace ground_truth{

std::vector<float> MaskedMM(
  mg_int H, mg_int D,
  const std::vector<mg_int>& row_offsets,
  const std::vector<mg_int>& column_indices,
  const std::vector<float>& vdata) {
  std::vector<float> ret(column_indices.size() * H, 0);
  for (size_t u = 0; u < row_offsets.size() - 1; u++) {
    for (mg_int eid = row_offsets[u]; eid < row_offsets[u + 1]; eid++) {
      mg_int v = column_indices[eid];
      for (mg_int idx = 0; idx < H * D; idx++)
        ret[eid * H + idx / D] += vdata[u * H * D + idx] * vdata[v * H * D + idx];
    }
  }
  return ret;
}

std::vector<float> Softmax(
  mg_int D,
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

} // end of namespace

#endif
