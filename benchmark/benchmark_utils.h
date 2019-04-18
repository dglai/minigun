#ifndef __BENCHMARK_BENCHMARK_UTILS_H_
#define __BENCHMARK_BENCHMARK_UTILS_H_

#include <time.h>
#include <utility>
#include <string>
#include <cstdlib>
#include <fstream>
#include "special_graph.h"

namespace utils{

#define CUDA_CALL(func)                                       \
  {                                                           \
    cudaError_t e = (func);                                   \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)  \
      << "CUDA: " << cudaGetErrorString(e);                   \
  }

template<typename T>
void VecPrint(const std::vector<T>& vec) {
  std::cout << "[";
  for (T t : vec) {
    std::cout << t << " ";
  }
  std::cout << "]" << std::endl;
}

template<typename T>
bool VecAllClose(const std::vector<T>& v1,
    const std::vector<T>& v2) {
  if (v1.size() != v2.size()) return false;
  for (size_t i = 0; i < v1.size(); ++i) {
    if (fabs(v1[i] - v2[i]) >= 1e-5 * fabs(v1[i])) {
      std::cout << "@" << i << ": " << v1[i]
                << " v.s. " << v2[i] << std::endl;
      return false;
    }
  }
  return true;
}

template<typename T>
void LoadGraph(const std::string& path,
    T& N,
    std::vector<std::pair<T, T> >& edges) {
  std::ifstream fin(path);
  if (!fin) {
    // TODO: throw an error here.
    exit(1);
  }
  T M;
  fin >> N >> M;
  for (T i = 0; i < M; i++) {
    T src, dst;
    fin >> src >> dst;
    edges.push_back(std::make_pair(src, dst));
  }
}

template<typename T>
void Coo2Csr(const T N,
    std::vector<std::pair<T, T> > edges,
    std::vector<T>& row_offsets,
    std::vector<T>& column_indices) {
  std::sort(edges.begin(), edges.end(), [](const std::pair<T, T> &e1, const std::pair<T, T> &e2) {
      if (e1.first == e2.first)
        return e1.second < e2.second;
      else
        return e1.first < e2.first;
    });
  row_offsets.resize(N + 1, 0);
  for (auto &e: edges) {
    T src = e.first, dst = e.second;
    row_offsets[src + 1]++;
    column_indices.push_back(dst);
  }
  for (T i = 1; i < N + 1; i++)
    row_offsets[i] = row_offsets[i - 1] + row_offsets[i];
}

template<typename T>
void CsrTranspose(const std::vector<T>& row_offsets,
    const std::vector<T>& column_indices,
    std::vector<T>& row_offsets_T,
    std::vector<T>& column_indices_T) {
  std::vector<std::pair<T, T> > edges;
  T N = row_offsets.size() - 1;
  for (T i = 0; i < N; i++) {
    for (T j = row_offsets[i]; j < row_offsets[i + 1]; j++) {
      T u = i, v = column_indices[j];
      edges.push_back(std::make_pair(v, u));
    }
  }
  Coo2Csr(N, edges, row_offsets_T, column_indices_T);
}

// Find the number of threads that is:
// - power of two
// - smaller or equal to dim
// - smaller or equal to max_nthrs
__inline__ int _FindNumThreads(int dim, int max_nthrs) {
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

__inline__ void CreateSparseBatch0(int64_t N,
    std::vector<mg_int>& row_offsets,
    std::vector<mg_int>& column_indices) {
  return graph::segtree_transformer_csr<mg_int>(N, 20, 40, false, row_offsets, column_indices);
}

__inline__ void CreateSparseBatch1(int64_t N,
    std::vector<mg_int>& row_offsets,
    std::vector<mg_int>& column_indices) {
  return graph::segtree_transformer_csr<mg_int>(N, 512, 512, false, row_offsets, column_indices);
}

__inline__ void CreateFullBatch0(int64_t N,
    std::vector<mg_int>& row_offsets,
    std::vector<mg_int>& column_indices) {
  return graph::full_transformer_csr<mg_int>(N, 20, 40, false, row_offsets, column_indices);
}

__inline__ void CreateFullBatch1(int64_t N,
    std::vector<mg_int>& row_offsets,
    std::vector<mg_int>& column_indices) {
  return graph::full_transformer_csr<mg_int>(N, 512, 512, false, row_offsets, column_indices);
}

__inline__ void CreateDatasetGraph(const std::string& path,
    std::vector<mg_int>& row_offsets,
    std::vector<mg_int>& column_indices) {
  mg_int N = 0;
  std::vector<std::pair<mg_int, mg_int> > edges;
  LoadGraph<mg_int>(path, N, edges);
  Coo2Csr(N, edges, row_offsets, column_indices);
}

} // namespace utils

#endif
