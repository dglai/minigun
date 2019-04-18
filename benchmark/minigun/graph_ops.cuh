#ifndef __MINIGUN_OPS_H
#define __MINIGUN_OPS_H

// TODO: template <typename scalar_t>
#include <cuda_runtime.h>
#include <cstdio>


namespace functor{

__device__ __forceinline__ float MyAtomicMax(float* addr, float val) {
  uint32_t* addr_as_ui = reinterpret_cast<uint32_t*>(addr);
  uint32_t old = *addr_as_ui;
  uint32_t assumed = old;
  do {
    assumed = old;
    old = atomicCAS(addr_as_ui, assumed,
        __float_as_uint(fmax(val, __uint_as_float(old))));
  } while (assumed != old);
  return __uint_as_float(old);
}

// Max
template <typename GData>
struct EdgeMax {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    mg_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    mg_int stride_x = blockDim.x * gridDim.x;
    const mg_int dim = gdata->head;
    while (tx < dim) {
      MyAtomicMax(gdata->max + dst * dim + tx, gdata->score[eid * dim + tx]);
      tx += stride_x;
    }
  }
};

// minus max, exp and sum
template <typename GData>
struct MinuxMaxExpSum {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    mg_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    mg_int stride_x = blockDim.x * gridDim.x;
    const mg_int dim = gdata->head;
    while (tx < dim) {
      gdata->score[eid * dim + tx] = expf(
          gdata->score[eid * dim + tx] - gdata->max[dst * dim + tx]);
      atomicAdd(gdata->sum + dst * dim + tx, gdata->score[eid * dim + tx]);
      tx += stride_x;
    }
  }
};

// norm
template <typename GData>
struct Norm {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    mg_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    mg_int stride_x = blockDim.x * gridDim.x;
    const mg_int dim = gdata->head;
    while (tx < dim) {
      gdata->score[eid * dim + tx] /= gdata->sum[dst * dim + tx];
      tx += stride_x;
    }
  }
};

// masked mm
template <typename GData>
struct MaskedMM {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  // ndata: (n, h, d)
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    
    // only one block along the data dim
    // TODO: zero init
    mg_int tx = threadIdx.x;
    const mg_int dim = gdata->dim * gdata->head;
    while (tx < dim) {
      atomicAdd(gdata->score + eid * gdata->head + (tx / gdata->dim),
                gdata->ndata[src * dim + tx] * gdata->ndata[dst * dim + tx]);
      tx += blockDim.x;
    }
  }
};

// Vector SPMM
template <typename GData>
struct VecSPMM {
  static __device__ __forceinline__ bool CondEdge(
    mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  // ndata: (n, h, d)
  // edata: (e, h)
  static __device__ __forceinline__ void ApplyEdge(
    mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    const int D = gdata->D;
    const int H = gdata->H;
    // each thread handles one attention head
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    while (h < H) {
      for (int i = 0; i < D; i++) {
        atomicAdd(gdata->feat_y + (src * H + h) * D + i,
                  gdata->feat_x[(src * H + h) * D + i] * gdata->weight[eid * H + h]);
      }
      h += blockDim.x;
    }
  }
};

// backward masked mm
template <typename GData>
struct BackMaskedMM {
  static __device__ __forceinline__ bool CondEdge(
    mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
    mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    const int D = gdata->D;
    const int H = gdata->H;
    // each thread handles one attention head
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    while (h < H) {
      for (int i = 0; i < D; i++) {
        atomicAdd(gdata->grad_x + (src * H + h) * D + i, gdata->grad_weight[eid * H + h] * gdata->feat_y[(dst * H + h) * D + i]);
        atomicAdd(gdata->grad_y + (dst * H + h) * D + i, gdata->grad_weight[eid * H + h] * gdata->feat_x[(src * H + h) * D + i]);
      }
      h += blockDim.x;
    }
  }
};

// backward softmax phase 0
template <typename GData>
struct BackSoftmaxAccum {
  static __device__ __forceinline__ bool CondEdge(
    mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  // accum: (N, H)
  static __device__ __forceinline__ void ApplyEdge(
    mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    const int H = gdata->H;
    // each thread handles one attention head
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    while (h < H) {
      sds = gdata->score[eid * H + h] * gdata->grad_score[eid * H + h];
      gdata->grad_weight[eid * H + h] = sds;
      atomicAdd(gdata->accum[dst * H + h], sds);
      h += blockDim.x;
    }
  }
}

template <typename GData>
struct BackSoftmaxMinus {
  static __device__ __forceinline__ bool CondEdge(
    mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  // accum: (N, H)
  static __device__ __forceinline__ void ApplyEdge(
    mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    const int H = gdata->H;
    // each thread handles one attention head
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    while (h < H) {
      gdata->grad_weight[eid * H + h] -= gdata->score[eid * H + h] * gdata->accum[dst * H + h];
      h += blockDim.x;
    }
  }
}

// backward vector spmm
template <typename GData>
struct BackVecSPMM {
  static __device__ __forceinline__ bool CondEdge(
    mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
    mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    // only one block along the data dim
    mg_int tx = threadIdx.x;
    const int D = gdata->D;
    const int H = gdata->H;
    // each thread handles one attention head
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    while (h < H) {
      float sum = 0;
      for (int i = 0; i < D; i++) {
        sum += gdata->feat_x[(src * H + h) * D + i] * gdata->grad_y[(dst * H + h) * D + i];
        atomicAdd(gdata->grad_x + (src * H + h) * D + i, gdata->weight[eid * H + h] * gdata->grad_y[(dst * H + h) * D + i]);
      }
      gdata->grad_weight[eid * H + h] = sum;
      h += blockDim.x;
    }
  }
}

} // end of namespace

#endif
