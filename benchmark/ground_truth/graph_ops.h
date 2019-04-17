#ifndef __GROUND_TRUTH_OPS_H
#define __GROUND_TRUTH_OPS_H

#include <iostream>

namespace ground_truth{

std::vector<float> MaskedMM(
  mg_int H, mg_int D,
  const std::vector<mg_int>& row_offsets,
  const std::vector<mg_int>& column_indices,
  const std::vector<float>& feat_x,
  const std::vector<float>& feat_y) {
  std::vector<float> ret(column_indices.size() * H, 0);
  for (size_t u = 0; u < row_offsets.size() - 1; u++) {
    for (mg_int eid = row_offsets[u]; eid < row_offsets[u + 1]; eid++) {
      mg_int v = column_indices[eid];
      for (mg_int idx = 0; idx < H * D; idx++)
        ret[eid * H + idx / D] += feat_x[u * H * D + idx] * feat_y[v * H * D + idx];
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

std::vector<float> VecSPMM(
  mg_int H,
  mg_int D,
  const std::vector<mg_int>& row_offsets,
  const std::vector<mg_int>& column_indices,
  const std::vector<float>& feat_x,
  const std::vector<float>& weight) {
  const size_t N = row_offsets.size() - 1;
  std::vector<float> ret(N * H * D, 0.);
  for (size_t u = 0; u < row_offsets.size() - 1; u++) {
    for (mg_int eid = row_offsets[u]; eid < row_offsets[u + 1]; eid++) {
      mg_int v = column_indices[eid];
      for (mg_int h = 0; h < H; h++) {
        for (mg_int idx = 0; idx < D; idx++)
          ret[((v * H) + h) * D + idx] += feat_x[((u * H) + h) * D + idx] * weight[eid];
      }
    }
}

std::vector<float> BackSoftmax(
  mg_int D,
  const std::vector<mg_int>& row_offsets,
  const std::vector<mg_int>& column_indices,
  const std::vector<float>& score,
  const std::vector<float>& grad_score) {
  const size_t N = row_offsets.size() - 1;
  std::vector<float> ret(score.size(), 0.);
  for (size_t u = 0; u < row_offsets.size() - 1; u++) {
    for (mg_int eid = row_offsets[u]; eid < row_offsets[u + 1]; eid++) {
      for (mg_int idx = 0; idx < D; h++) {
        float sum = 0;
        for (mg_int eid_ = row_offsets[u]; eid_ < row_offsets[u + 1]; eid++) {
          sum -= grad_score[eid_ * D + idx] * score[eid_ * D + idx] * score[eid * D + idx];
          if (eid == eid_)
            sum += grad_score[eid_ * D + idx] * score[eid_ * D + idx];
        }
        ret[eid * D + idx] = sum;
      }
    }
  }
  return ret;
}

std::pair<std::vector<float>, std::vector<float> > BackMaskedMM(
  mg_int H,
  mg_int D,
  const std::vector<mg_int>& row_offsets,
  const std::vector<mg_int>& column_indices,
  const std::vector<float>& feat_x,
  const std::vector<float>& feat_y,
  const std::vector<float>& grad_weight) {
  std::vector<float> grad_x(feat_x.size(), 0.);
  std::vector<float> grad_y(feat_y.size(), 0.);
  const size_t N = row_offsets.size() - 1;
  for (size_t u = 0; u < row_offsets.size() - 1; u++) {
    for (mg_int eid = row_offsets[u]; eid < row_offsets[u + 1]; eid++) {
      mg_int v = column_indices[eid];
      for (mg_int h = 0; h < H; h++)
        for (mg_int idx = 0; idx < D; idx++) {
          grad_x[(u * H + h) * D + idx] += grad_weight[eid * H + h] * feat_y[(v * H + h) * D + idx];
          grad_y[(v * H + h) * D + idx] += grad_weight[eid * H + h] * feat_x[(u * H + h) * D + idx];
        }
    }
  }
  return std::make_pair(grad_x, grad_y);
}

std::pair<std::vector<float>, std::vector<float> > BackVecSPMM(
  mg_int H,
  mg_int D,
  const std::vector<mg_int>& row_offsets,
  const std::vector<mg_int>& column_indices,
  const std::vector<float>& feat_x,
  const std::vector<float>& weight,
  const std::vector<float>& grad_y) {
  std::vector<float> grad_x(feat_x.size(), 0.);
  std::vector<float> grad_weight(weight.size(), 0.);
  const size_t N = row_offsets.size() - 1;
  for (size_t u = 0; u < row_offsets.size() - 1; u++) {
    for (mg_int eid = row_offsets[u]; eid < row_offsets[u + 1]; eid++) {
      mg_int v = column_indices[eid];
      for (mg_int h = 0; h < H; h++) {
        float sum = 0;
        for (mg_int idx = 0; idx < D; idx++) {
          grad_x[(u * H + h) * D + idx] += weight[eid * H + h] * feat_y[(v * H + h) * D + idx];
          sum += feat_x[(u * H + h) * D + idx] * grad_y[(v * H + h) * D + idx];
        }
        grad_weight[eid * H + h] = sum;
      }
    }
  }
  return std::make_pair(grad_x, grad_weight);
}

} // end of namespace


#endif
