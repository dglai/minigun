#ifndef SAMPLES_SAMPLES_UTILS_H_
#define SAMPLES_SAMPLES_UTILS_H_

#include <time.h>

namespace utils {

#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
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
bool VecEqual(const std::vector<T>& v1,
           const std::vector<T>& v2) {
  if (v1.size() != v2.size()) return false;
  for (size_t i = 0; i < v1.size(); ++i) {
    if (fabs(v1[i] - v2[i]) >= 1e-5 * fabs(v1[i]) + 1e-4) {
      std::cout << "@" << i << ": " << v1[i]
        << " v.s. " << v2[i] << std::endl;
      return false;
    }
  }
  return true;
}

__inline__ void CreateNPGraph(int64_t N, float P,
    std::vector<mg_int>& row_offsets,
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

// Find the number of threads that is:
//  - power of two 
//  - smaller or equal to dim
//  - smaller or equal to max_nthrs
__inline__ int _FindNumThreads(int dim, int max_nthrs) {
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

}  // namespace utils

#endif  // SAMPLES_SAMPLES_UTILS_H_
