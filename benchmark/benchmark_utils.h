#ifndef __BENCHMARK_BENCHMARK_UTILS_H_
#define __BENCHMAKR_BENCHMARK_UTILS_H_

#include <time.h>
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

} // namespace utils

#endif
