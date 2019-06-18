#ifndef SAMPLES_SAMPLES_UTILS_H_
#define SAMPLES_SAMPLES_UTILS_H_

#include <vector>
#include <map>
#include <cmath>
#include <ctime>
#include "sys/time.h"

#include <dmlc/io.h>

namespace utils {

#define FCLOSE(a, b) (fabs((a) - (b)) < 1e-5 * fabs(a) + 1e-4)

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
    if (!(FCLOSE(v1[i], v2[i]))) {
      return false;
      //std::cout << "@" << i << ": " << v1[i] << " v.s. " << v2[i] << std::endl;
    }
  }
  return true;
}

template <typename Iter1, typename Iter2>
bool IterEqual(Iter1 v1, Iter2 v2, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!(FCLOSE(v1[i], v2[i]))) {
      return false;
      //std::cout << "@" << i << ": " << v1[i] << " v.s. " << v2[i] << std::endl;
    }
  }
  return true;
}

inline void CreateNPGraph(int64_t N, float P,
    std::vector<int32_t>& row_offsets,
    std::vector<int32_t>& column_indices) {
  row_offsets.resize(N+1, 0);
  row_offsets[0] = 0;
  for (int32_t u = 0; u < N; ++u) {
    for (int32_t v = 0; v < N; ++v) {
      if ((float)rand() / RAND_MAX < P) {
        column_indices.push_back(v);
      }
    }
    row_offsets[u + 1] = column_indices.size();
  }
}

struct CPUAllocator {

  template <typename T>
  T* AllocateData(size_t bytes) {
    void* ptr = malloc(bytes);
    return static_cast<T*>(ptr);
  }

  void FreeData(void* ptr) {
    free(ptr);
  }

  template <typename T>
  T* AllocateWorkspace(size_t bytes) {
    void* ptr;
    if (!pool[bytes].empty()) {
      ptr = pool[bytes].back();
      pool[bytes].pop_back();
    } else {
      ptr = malloc(bytes);
    }
    wspace_size[ptr] = bytes;
    return static_cast<T*>(ptr);
  }

  void FreeWorkspace(void* ptr) {
    assert(wspace_size.count(ptr));
    pool[wspace_size[ptr]].push_back(ptr);
  }

  std::map<void*, size_t> wspace_size;
  std::map<size_t, std::vector<void*>> pool;

  static CPUAllocator* Get() {
    static CPUAllocator alloc;
    return &alloc;
  }
};

#ifdef __CUDACC__
#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
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

struct GPUAllocator {

  template <typename T>
  T* AllocateData(size_t bytes) {
    void* ptr;
    CUDA_CALL(cudaMalloc(&ptr, bytes));
    return static_cast<T*>(ptr);
  }

  void FreeData(void* ptr) {
    CUDA_CALL(cudaFree(ptr));
  }

  template <typename T>
  T* AllocateWorkspace(size_t bytes) {
    void* ptr;
    if (!pool[bytes].empty()) {
      ptr = pool[bytes].back();
      pool[bytes].pop_back();
    } else {
      CUDA_CALL(cudaMalloc(&ptr, bytes));
    }
    wspace_size[ptr] = bytes;
    return static_cast<T*>(ptr);
  }

  void FreeWorkspace(void* ptr) {
    assert(wspace_size.count(ptr));
    pool[wspace_size[ptr]].push_back(ptr);
  }

  std::map<void*, size_t> wspace_size;
  std::map<size_t, std::vector<void*>> pool;

  static GPUAllocator* Get() {
    static GPUAllocator alloc;
    return &alloc;
  }
};

#endif  // __CUDACC__

}  // namespace utils

#endif  // SAMPLES_SAMPLES_UTILS_H_
