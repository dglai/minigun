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

// Cuda context that is compatible with modern gpu
// One should aware that there might be some overhead to construct this context.
class MgpuContext : public mgpu::context_t {
 public:
  MgpuContext(cudaStream_t stream): stream_(stream) {
    cudaFuncAttributes attr;
    CUDA_CALL(cudaFuncGetAttributes(&attr, dummy_k<0>));
    ptx_version_ = attr.ptxVersion;

    int ord;
    CUDA_CALL(cudaGetDevice(&ord));
    CUDA_CALL(cudaGetDeviceProperties(&props_, ord));
    CUDA_CALL(cudaEventCreate(&event_));
  }
  ~MgpuContext() {
    CUDA_CALL(cudaEventDestroy(event_));
  }
  const cudaDeviceProp& props() const override {
    return props_;
  } 
  int ptx_version() const override {
    return ptx_version_;
  }
  cudaStream_t stream() override {
    return stream_;
  }
  void* alloc(size_t size, mgpu::memory_space_t space) override {
    LOG(FATAL) << "Alloc is not implemented.";
    return nullptr;
  }
  void free(void* p, mgpu::memory_space_t space) override {
    LOG(FATAL) << "Free is not implemented.";
  }
  void synchronize() override {
    if (stream_) {
      CUDA_CALL(cudaStreamSynchronize(stream_));
    } else {
      CUDA_CALL(cudaDeviceSynchronize());
    }
  }
  cudaEvent_t event() override {
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
  int ptx_version_;
  cudaStream_t stream_;
  cudaEvent_t event_;
  cudaDeviceProp props_;
};

}  // namespace minigun
#endif  // MINIGUN_CUDA_CUDA_COMMON_H_
