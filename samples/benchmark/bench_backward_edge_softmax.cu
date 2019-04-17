#include <iostream>
#include <cstdlib>
#include <limits>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "./baseline/yzh_kernels.cuh"
#include "../samples_io.h"
#include "../samples_utils.h"

using minigun::advance::RuntimeConfig;

struct GData {
  int H = 0;  // num heads
  float* score{nullptr};
  float* grad_score{nullptr};
  float* accum{nullptr};
  float* out{nullptr};
};

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

// backward softmax phase 0
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
    float* score_off = gdata->score + eid * H;
    float* grad_score_off = gdata->grad_score + eid * H;
    float* accum_off = gdata->accum + dst * H;
    float* ret_off = gdata->out + eid * H;
    while (h < H) {
      float sds = __ldg(score_off + h) * __ldg(grad_score_off + h);
      atomicAdd(accum_off + h, sds);
      *(ret_off + h) = sds;
      h += blockDim.x;
    }
  }
};

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
    float* score_off = gdata->score + eid * H;
    float* accum_off = gdata->accum + dst * H;
    float* ret_off = gdata->out + eid * H;
    while (h < H) {
      *(ret_off + h) -= __ldg(score_off + h) * __ldg(accum_off + h);
      h += blockDim.x;
    }
  }
};


void InitGData(GData* gdata, mg_int N, mg_int M) {
  std::vector<float> accum(N * gdata->H, 0.);
  std::vector<float> score(M * gdata->H, 0.), grad_score(M * gdata->H, 0.), out(M * gdata->H, 0.);
  for (mg_int i = 0; i < score.size(); ++i) {
    score[i] = (float)rand() / RAND_MAX;
    grad_score[i] = (float)rand() / RAND_MAX;
  }
  CUDA_CALL(cudaMalloc(&(gdata->accum), sizeof(float) * accum.size()));
  CUDA_CALL(cudaMemcpy(gdata->accum, &accum[0],
        sizeof(float) * accum.size(), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->out), sizeof(float) * out.size()));
  CUDA_CALL(cudaMemcpy(gdata->out, &out[0],
        sizeof(float) * out.size(), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->score), sizeof(float) * score.size()));
  CUDA_CALL(cudaMemcpy(gdata->score, &score[0],
        sizeof(float) * score.size(), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->grad_score), sizeof(float) * grad_score.size()));
  CUDA_CALL(cudaMemcpy(gdata->grad_score, &grad_score[0],
        sizeof(float) * grad_score.size(), cudaMemcpyHostToDevice));
}

double RunMinigun(const RuntimeConfig& rtcfg, const minigun::Csr& csr, GData* d_gdata) {
  minigun::IntArray1D infront, outfront;

  // dry run
  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, Config, GData, BackSoftmaxAccum>(
      rtcfg, csr, d_gdata, infront, outfront);
  minigun::advance::Advance<kDLGPU, Config, GData, BackSoftmaxMinus>(
      rtcfg, csr, d_gdata, infront, outfront);
  CUDA_CALL(cudaDeviceSynchronize());

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLGPU, Config, GData, BackSoftmaxAccum>(
        rtcfg, csr, d_gdata, infront, outfront);
    minigun::advance::Advance<kDLGPU, Config, GData, BackSoftmaxMinus>(
        rtcfg, csr, d_gdata, infront, outfront);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  gettimeofday(&t1, nullptr);
  double dur = (double)(t1.tv_sec * 1e6 + t1.tv_usec -
      (t0.tv_sec * 1e6 + t0.tv_usec)) / K / 1000.0;  // ms

  return dur;
}

double RunBaseline1(const RuntimeConfig& rtcfg, const minigun::Csr& csr, GData* gdata) {
  const mg_int N = csr.row_offsets.length - 1;
  const int H = gdata->H;

  if (H > 64) {
    // thread block can have at most 64 threads in z dimension.
    return 100.0;
  }

  // dry run
  custom_kernel::sparse_softmax_backward_kernel_no_eid<mg_int, float>
    <<<dim3(N, 1, 1), dim3(1, 256/H, H)>>>(
      csr.row_offsets.data,
      gdata->grad_score,
      gdata->score,
      gdata->out,
      (int)N, (int)H);
  CUDA_CALL(cudaDeviceSynchronize());

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    custom_kernel::sparse_softmax_backward_kernel_no_eid<mg_int, float>
      <<<dim3(N, 1, 1), dim3(1, 256/H, H)>>>(
        csr.row_offsets.data,
        gdata->grad_score,
        gdata->score,
        gdata->out,
        (int)N, (int)H);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  gettimeofday(&t1, nullptr);
  double dur = (double)(t1.tv_sec * 1e6 + t1.tv_usec -
      (t0.tv_sec * 1e6 + t0.tv_usec)) / K / 1000.0;  // ms

  return dur;
}

int main(int argc, char** argv) {
  srand(42);
  if (argc < 3) {
    std::cout << "USAGE: ./bench_backward_edge_softmax <file_name> <num_heads>" << std::endl;
    return 1;
  }
  const char* filename = argv[1];
  const int num_heads = std::atoi(argv[2]);
  std::cout << "filename=" << filename << " num_heads=" << num_heads << std::endl;

  utils::SampleCsr scsr;
  utils::LoadGraphFromFile(filename, &scsr);
  const mg_int N = scsr.row_offsets.size() - 1;
  const mg_int M = scsr.column_indices.size();
  std::cout << "#Nodes: " << N << " #Edges: " << M << std::endl;

  // csr
  minigun::Csr csr = utils::ToMinigunCsr(scsr, kDLGPU);

  // gdata
  GData gdata;
  gdata.H = num_heads;
  InitGData(&gdata, N, M);
  GData* d_gdata;
  CUDA_CALL(cudaMalloc(&d_gdata, sizeof(GData)));
  CUDA_CALL(cudaMemcpy(d_gdata, &gdata, sizeof(GData), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaDeviceSynchronize());
  
  // create stream
  RuntimeConfig cfg;
  cfg.ctx = {kDLGPU, 0};
  int nt = utils::_FindNumThreads(gdata.H, 32);
  cfg.data_num_threads = nt;
  cfg.data_num_blocks = gdata.H / nt;
  CUDA_CALL(cudaStreamCreate(&cfg.stream));

  double dur1 = RunMinigun(cfg, csr, d_gdata);
  std::cout << "minigun time(ms): " << dur1 << std::endl;
  double dur2 = RunBaseline1(cfg, csr, &gdata);
  std::cout << "baseline1 time(ms): " << dur2 << std::endl;

  return 0;
}
