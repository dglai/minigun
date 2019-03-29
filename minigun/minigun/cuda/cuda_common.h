/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef MINIGUN_CUDA_CUDA_COMMON_H_
#define MINIGUN_CUDA_CUDA_COMMON_H_

#include <cuda_runtime.h>
#include <string>

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

}  // namespace minigun
#endif  // MINIGUN_CUDA_CUDA_COMMON_H_
