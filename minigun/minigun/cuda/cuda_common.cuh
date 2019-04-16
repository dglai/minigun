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
#include <moderngpu/context.hxx>

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

// Cuda context that is compatible with modern gpu
template <typename Alloc>
class MgpuContext : public mgpu::context_t {
 public:
  MgpuContext(int device_id, cudaStream_t stream, Alloc* alloc):
    cuda_ctx_(CudaContext::Get(device_id)),
    stream_(stream), alloc_(alloc) {
    //CUDA_CALL(cudaEventCreate(&event_));
  }
  ~MgpuContext() {
    //CUDA_CALL(cudaEventDestroy(event_));
  }
  const cudaDeviceProp& props() const override {
    return cuda_ctx_.props();
  } 
  int ptx_version() const override {
    return cuda_ctx_.ptx_version();
  }
  cudaStream_t stream() override {
    return stream_;
  }
  void* alloc(size_t size, mgpu::memory_space_t space) override {
    CHECK_EQ(space,  mgpu::memory_space_device);
    return alloc_->template AllocateWorkspace<void>(size);
  }
  void free(void* p, mgpu::memory_space_t space) override {
    CHECK_EQ(space,  mgpu::memory_space_device);
    alloc_->FreeWorkspace(p);
  }
  void synchronize() override {
    if (stream_) {
      CUDA_CALL(cudaStreamSynchronize(stream_));
    } else {
      CUDA_CALL(cudaDeviceSynchronize());
    }
  }
  cudaEvent_t event() override {
    LOG(FATAL) << "event is not implemented.";
    return event_;
  }
  void timer_begin() override {
    LOG(FATAL) << "timer_begin is not implemented.";
  }
  double timer_end() override {
    LOG(FATAL) << "timer_end is not implemented.";
    return 0.0;
  }

 private:
  const CudaContext& cuda_ctx_;
  cudaStream_t stream_;
  Alloc* alloc_;
  cudaEvent_t event_;
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
