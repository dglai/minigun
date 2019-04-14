#ifndef MINIGUN_MEM_H_
#define MINIGUN_MEM_H_

#include <dmlc/logging.h>

#ifdef MINIGUN_USE_CUDA
#include "./cuda/cuda_common.cuh"
#endif  // MINIGUN_USE_CUDA

namespace minigun {

template <int XPU>
class DefaultAllocator {
 public:
  static DefaultAllocator<XPU>* Get() {
    LOG(FATAL) << "Unsupported device type: " << XPU;
    return nullptr;
  }
};

template <>
class DefaultAllocator<kDLCPU> {
 public:
  template <typename T>
  T* AllocateData(size_t bytes) {
    return static_cast<T*>(malloc(bytes));
  }
  template <typename T>
  T* AllocateWorkspace(size_t bytes) {
    return static_cast<T*>(malloc(bytes));
  }
  void FreeData(void* ptr) {
    free(ptr);
  }
  void FreeWorkspace(void* ptr) {
    free(ptr);
  }
  static DefaultAllocator<kDLCPU>* Get() {
    static DefaultAllocator<kDLCPU> alloc;
    return &alloc;
  }
};

#ifdef MINIGUN_USE_CUDA
template <>
class DefaultAllocator<kDLGPU> {
 public:
  template <typename T>
  T* AllocateData(size_t bytes) {
    void* ptr;
    CUDA_CALL(cudaMalloc(&ptr, bytes));
    return static_cast<T*>(ptr);
  }
  template <typename T>
  T* AllocateWorkspace(size_t bytes) {
    void* ptr;
    CUDA_CALL(cudaMalloc(&ptr, bytes));
    return static_cast<T*>(ptr);
  }
  void FreeData(void* ptr) {
    CUDA_CALL(cudaFree(ptr));
  }
  void FreeWorkspace(void* ptr) {
    CUDA_CALL(cudaFree(ptr));
  }
  static DefaultAllocator<kDLGPU>* Get() {
    static DefaultAllocator<kDLGPU> alloc;
    return &alloc;
  }
};
#endif  // MINIGUN_USE_CUDA

}  // namespace minigun

#endif  // MINIGUN_MEM_H_
