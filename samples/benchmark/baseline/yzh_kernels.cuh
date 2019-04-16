#ifndef SAMPLES_BENCHMARK_BASELINE_YZH_KERNELS_CUH_
#define SAMPLES_BENCHMARK_BASELINE_YZH_KERNELS_CUH_

namespace custom_kernel{

/*
 * CUDA Kernel of the forward function for Masked Matrix Multiplication. (argument: csr format)
 */
template <typename idx_t, typename data_t>
__global__ void maskedmm_csr_forward_kernel(
    const idx_t* __restrict__ indptr,
    const idx_t* __restrict__ indices,
    const data_t* __restrict__ A,
    const data_t* __restrict__ B,
    data_t* __restrict__ y,
    const int d, const int n, const int h) {
  int i = blockIdx.x; 
  int tx = threadIdx.x;

  if (i < n) {
    for (idx_t j = indptr[i] + tx; j < indptr[i + 1]; j += blockDim.x) {
      for (int ko = 0; ko < h; ++ko) {
        data_t sum = 0;
        for (int ki = 0; ki < d; ++ki) {
          sum += A[(i * h + ko) * d + ki] * B[(indices[j] * h + ko) * d + ki]; //B[(ko * d + ki) * n + indices[j]];
        }
        y[j * h + ko] = sum;
      }
    }
  }
}

/*
 * CUDA Kernel of the backward function for Masked Matrix Multiplication. (argument: csr format)
 */
template <typename idx_t, typename data_t>
__global__ void maskedmm_csr_backward_kernel(
    const idx_t* __restrict__ indptr_r,
    const idx_t* __restrict__ eid_r,
    const idx_t* __restrict__ indices_r,
    const idx_t* __restrict__ indptr_c,
    const idx_t* __restrict__ eid_c,
    const idx_t* __restrict__ indices_c,
    const data_t* __restrict__ A,
    const data_t* __restrict__ B,
    const data_t* __restrict__ dy,
    data_t* __restrict__ dA,
    data_t* __restrict__ dB,
    const int d, const int n, const int h) {
  int tx = threadIdx.x;
  int i = blockIdx.x;
  if (i < n) {
    for (int j = tx; j < d * h; j += blockDim.x) {
      data_t sum = 0;
      for (int k = indptr_r[i]; k < indptr_r[i + 1]; ++k)
        sum += dy[eid_r[k] * h + j / d] * B[indices_r[k] * d * h + j];
      dA[i * d * h + j] = sum;

      sum = 0;
      for (int k = indptr_c[i]; k < indptr_c[i + 1]; ++k)
        sum += dy[eid_c[k] * h + j / d] * A[indices_c[k] * d * h + j];
      dB[i * d * h + j] = sum;
    }
  }
}

/*
 * CUDA Kernel of the forward function for Source Multiply Edge Function.
 * For `src_mul_edge` operation, the arguments are csr(column-major) representations.
 */
template <typename idx_t, typename data_t>
__global__ void vector_spmm_forward_kernel(
    const idx_t* __restrict__ indptr,
    const idx_t* __restrict__ eid,
    const idx_t* __restrict__ indices,
    const data_t* __restrict__ edata,
    const data_t* __restrict__ x,
    data_t* __restrict__ y,
    const int d, const int n, const int h) {
  int i = blockIdx.x;
  int tx = threadIdx.x;
  if (i < n) {
    for (int j = tx; j < d * h; j += blockDim.x) {
      data_t sum = 0;
      for (int k = indptr[i]; k < indptr[i + 1]; ++k)
        sum += edata[eid[k] * h + j / d] * x[indices[k] * d * h + j];
      y[i * d * h + j] = sum;
    }
  }
}

/*
 * CUDA Kernel of the forward function for Source Multiply Edge Function.
 * For `src_mul_edge` operation, the arguments are csr(column-major) representations.
 * no eid
 */
template <typename idx_t, typename data_t>
__global__ void vector_spmm_forward_kernel_no_eid(
    const idx_t* __restrict__ indptr,
    const idx_t* __restrict__ indices,
    const data_t* __restrict__ edata,
    const data_t* __restrict__ x,
    data_t* __restrict__ y,
    const int d, const int n, const int h) {
  int i = blockIdx.x;
  int tx = threadIdx.x;
  if (i < n) {
    for (int j = tx; j < d * h; j += blockDim.x) {
      data_t sum = 0;
      for (int k = indptr[i]; k < indptr[i + 1]; ++k)
        sum += edata[h + j / d] * x[indices[k] * d * h + j];
      y[i * d * h + j] = sum;
    }
  }
}

/*
 * CUDA Kernel of the backward function for Source Multiply Edge Function.
 */
template <typename idx_t, typename data_t>
__global__ void vector_spmm_backward_kernel_0(
    const idx_t* __restrict__ indptr,
    const idx_t* __restrict__ eid,
    const idx_t* __restrict__ indices,
    const data_t* __restrict__ dy,
    const data_t* __restrict__ xt,
    data_t* __restrict__ dedata,
    const int d, const int n, const int h) {
  int i = blockIdx.x; 
  int tx = threadIdx.x;
  if (i < n) {
    for (int j = indptr[i] + tx; j < indptr[i + 1]; j += blockDim.x)
      for (int ko = 0; ko < h; ++ko) {
        data_t sum = 0;
        for (int ki = 0; ki < d; ++ki) {
          sum += dy[(i * h + ko) * d + ki] * xt[(ko * d + ki) * n + indices[j]];
        }
        dedata[eid[j] * h + ko] = sum;
      }
  }
}

template <typename idx_t, typename data_t>
__global__ void vector_spmm_backward_kernel_1(
    const idx_t* __restrict__ indptr,
    const idx_t* __restrict__ eid,
    const idx_t* __restrict__ indices,
    const data_t* __restrict__ edata,
    const data_t* __restrict__ dy,
    data_t* __restrict__ dx,
    const int d, const int n, const int h) {
  int i = blockIdx.x; 
  int tx = threadIdx.x;
  if (i < n) {
    for (int j = tx; j < d * h; j += blockDim.x) {
      data_t sum = 0;
      for (int k = indptr[i]; k < indptr[i + 1]; ++k)
        sum += edata[eid[k] * h + j / d] * dy[indices[k] * d * h + j];
      dx[i * d * h + j] = sum;
    }
  }
}

/*
 * CUDA Kernel of forward function for Sparse Softmax
 * y = softmax(x), grouped by node.
 * indptr, eid: csr format
 */
template <typename idx_t, typename data_t>
__global__ void sparse_softmax_forward_kernel(
    const idx_t* __restrict__ indptr,
    const data_t* __restrict__ x,
    data_t* __restrict__ y,
    const int n, const int h) {
  //data_t max_val = -1e9;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = threadIdx.y;
  if (i < n) {
    data_t max_val = (indptr[i] < indptr[i + 1]) ? x[indptr[i] * h + j]: (data_t)(0);
    for (int k = indptr[i]; k < indptr[i + 1]; ++k)
      max_val = max(max_val, x[k * h + j]);

    data_t sum = 0;
    for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
      data_t now = exp(x[k * h + j] - max_val);
      y[k * h + j] = now;
      sum += now;
    }

    for (int k = indptr[i]; k < indptr[i + 1]; ++k)
      y[k * h + j] /= sum;
  }
}

/*
 * CUDA Kernel of backward function for Sparse Softmax.
 * indptr, eid: csr format
 */
template <typename idx_t, typename data_t>
__global__ void sparse_softmax_backward_kernel(
    const idx_t* __restrict__ indptr,
    const idx_t* __restrict__ eid,
    const data_t* __restrict__ dy,
    const data_t* __restrict__ y,
    data_t* __restrict__ dx,
    const int n, const int h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  if (i < n) {
    for (int kj = indptr[i] + ty; kj < indptr[i + 1]; kj += blockDim.y) {
      data_t dsum = 0;
      for (int ki = indptr[i]; ki < indptr[i + 1]; ++ki) {
        dsum -= dy[eid[ki] * h + tz] * y[eid[ki] * h + tz] * y[eid[kj] * h + tz];
        if (ki == kj) dsum += dy[eid[ki] * h + tz] * y[eid[ki] * h + tz];
      }
      dx[eid[kj] * h + tz] = dsum;
    }
  }
}

} // end of namespace

#endif
