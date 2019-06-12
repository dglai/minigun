/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef MINIGUN_CUDA_CUDA_COMMON_H_
#define MINIGUN_CUDA_CUDA_COMMON_H_

#include <cuda_runtime.h>
#include <string>

#include <dmlc/logging.h>

namespace minigun {

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char *msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LOG(FATAL)                                                        \
          << "CUDAError: " #x " failed with error: " << msg;            \
    }                                                                   \
  }

#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

// Dummy kernel for retrieving PTX version.
template<int dummy_arg>
__global__ void dummy_k() { }

class CudaContext {
 public:
  explicit CudaContext(int device_id): device_id_(device_id) {
    cudaFuncAttributes attr;
    CUDA_CALL(cudaFuncGetAttributes(&attr, dummy_k<0>));
    ptx_version_ = attr.ptxVersion;
    CUDA_CALL(cudaGetDeviceProperties(&props_, device_id));
  }

  int device_id() const {
    return device_id_;
  }
  
  int ptx_version() const {
    return ptx_version_;
  }

  const cudaDeviceProp& props() const {
    return props_;
  }

  static const CudaContext& Get(int device_id) {
    static std::vector<CudaContext> contexts = InitContexts();
    return contexts[device_id];
  }

 private:
  // Init context for each device
  static std::vector<CudaContext> InitContexts() {
    std::vector<CudaContext> ret;
    int count = 0;
    CUDA_CALL(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
      ret.emplace_back(CudaContext{i});
    }
    return ret;
  }

  int device_id_;
  int ptx_version_;
  cudaDeviceProp props_;
};

template <typename T>
__device__ __forceinline__ T _ldg(T* addr) {
#if __CUDA_ARCH__ >= 350
  return __ldg(addr);
#else
  return *addr;
#endif
}

namespace types {

// Make sure unsupported type will not compile
extern __device__ __host__ void ErrorUnsupportedType();

template <typename T>
__device__ __host__ __forceinline__ T MaxValue() {
  ErrorUnsupportedType();
  return T();
}

template <>
__device__ __host__ __forceinline__ int32_t MaxValue<int32_t>() {
  return INT_MAX;
}

template <>
__device__ __host__ __forceinline__ int64_t MaxValue<int64_t>() {
  return LLONG_MAX;
}

}  // types

}  // namespace minigun
#endif  // MINIGUN_CUDA_CUDA_COMMON_H_
