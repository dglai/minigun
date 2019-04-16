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
  float* sum{nullptr};
  float* max{nullptr};
  float* ret{nullptr};
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

// Max
struct EdgeMax {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    const int H = gdata->H;
    float* inoff = gdata->score + eid * H;
    float* outoff = gdata->max + dst * H;
    while (tx < H) {
      MyAtomicMax(outoff + tx, __ldg(inoff + tx));
      tx += stride_x;
    }
  }
};

// minus max, exp and sum
struct MinusMaxExpSum {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    const int H = gdata->H;
    const float* score_off = gdata->score + eid * H;
    float* ret_off = gdata->ret + eid * H;
    float* max_off = gdata->max + dst * H;
    float* sum_off = gdata->sum + dst * H;
    while (tx < H) {
      const float new_score = expf(__ldg(score_off + tx) - __ldg(max_off + tx));
      atomicAdd(sum_off + tx, new_score);
      *(ret_off + tx) = new_score;
      tx += stride_x;
    }
  }
};

// norm
struct Norm {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    const int H = gdata->H;
    float* ret_off = gdata->ret + eid * H;
    float* sum_off = gdata->sum + dst * H;
    while (tx < H) {
      *(ret_off + tx) /= __ldg(sum_off + tx);
      tx += stride_x;
    }
  }
};

void InitGData(GData* gdata, mg_int N, mg_int M) {
  std::vector<float> sum(N * gdata->H, 0.), max(N * gdata->H, std::numeric_limits<float>::lowest());
  std::vector<float> score(M * gdata->H, 0.), ret(M * gdata->H, 0.);
  for (mg_int i = 0; i < score.size(); ++i) {
    score[i] = (float)rand() / RAND_MAX;
  }
  CUDA_CALL(cudaMalloc(&(gdata->sum), sizeof(float) * N * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->sum, &sum[0],
        sizeof(float) * N * gdata->H, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->max), sizeof(float) * N * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->max, &max[0],
        sizeof(float) * N * gdata->H, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->score), sizeof(float) * M * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->score, &score[0],
        sizeof(float) * M * gdata->H, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->ret), sizeof(float) * M * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->ret, &ret[0],
        sizeof(float) * M * gdata->H, cudaMemcpyHostToDevice));
}

double RunMinigun(const RuntimeConfig& rtcfg, const minigun::Csr& csr, GData* d_gdata) {
  minigun::IntArray1D infront, outfront;

  // dry run
  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, Config, GData, EdgeMax>(
      rtcfg, csr, d_gdata, infront, outfront);
  minigun::advance::Advance<kDLGPU, Config, GData, MinusMaxExpSum>(
      rtcfg, csr, d_gdata, infront, outfront);
  minigun::advance::Advance<kDLGPU, Config, GData, Norm>(
      rtcfg, csr, d_gdata, infront, outfront);
  CUDA_CALL(cudaDeviceSynchronize());

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLGPU, Config, GData, EdgeMax>(
        rtcfg, csr, d_gdata, infront, outfront);
    minigun::advance::Advance<kDLGPU, Config, GData, MinusMaxExpSum>(
        rtcfg, csr, d_gdata, infront, outfront);
    minigun::advance::Advance<kDLGPU, Config, GData, Norm>(
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

  // dry run
  custom_kernel::sparse_softmax_forward_kernel<mg_int, float><<<(N + 31) / 32, dim3(32, H)>>>(
      csr.row_offsets.data,
      gdata->score,
      gdata->ret,
      (int)N, (int)H);
  CUDA_CALL(cudaDeviceSynchronize());

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    custom_kernel::sparse_softmax_forward_kernel<mg_int, float><<<(N + 31) / 32, dim3(32, H)>>>(
        csr.row_offsets.data,
        gdata->score,
        gdata->ret,
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
    std::cout << "USAGE: ./bench_masked_mm <file_name> <num_heads>" << std::endl;
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
