#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "./baseline/yzh_kernels.cuh"
#include "../samples_io.h"
#include "../samples_utils.h"

using minigun::advance::RuntimeConfig;

struct GData {
  int D = 0;  // feat size
  int H = 0;  // num heads
  float* ndata{nullptr};
  float* score{nullptr};
};

struct MaskedMMFunctor {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData* gdata) {
    const int D = gdata->D;
    const int H = gdata->H;
    // each thread handles one attention head
    const mg_int h = blockIdx.x * blockDim.x + threadIdx.x;
    mg_int srcoff = src * (D * H) + h * D;
    mg_int dstoff = dst * (D * H) + h * D;
    const mg_int srcend = (src + 1) * (D * H);
    while (srcoff < srcend) {
      float sum = 0.;
      for (int i = 0; i < D; ++i) {
        sum += gdata->ndata[srcoff + i] * gdata->ndata[dstoff + i];
      }
      gdata->score[src * H + h] = sum;
      srcoff += blockDim.x * D;
      dstoff += blockDim.x * D;
    }
  }
};

void InitGData(GData* gdata, mg_int N) {
  std::vector<float> ndata(N * gdata->D * gdata->H), score(N * gdata->H, 0.);
  for (mg_int i = 0; i < ndata.size(); ++i) {
    ndata[i] = (float)rand() / RAND_MAX;
  }
  CUDA_CALL(cudaMalloc(&(gdata->ndata), sizeof(float) * N * gdata->D * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->ndata, &ndata[0],
        sizeof(float) * N * gdata->D * gdata->H, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->score), sizeof(float) * N * gdata->H));
  CUDA_CALL(cudaMemcpy(gdata->score, &score[0],
        sizeof(float) * N * gdata->H, cudaMemcpyHostToDevice));
}

double RunMinigun(const RuntimeConfig& rtcfg, const minigun::Csr& csr, GData* d_gdata) {
  minigun::IntArray1D infront, outfront;

  // dry run
  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, Config, GData, MaskedMMFunctor>(
      rtcfg, csr, d_gdata, infront, outfront);
  CUDA_CALL(cudaDeviceSynchronize());

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
    minigun::advance::Advance<kDLGPU, Config, GData, MaskedMMFunctor>(
        rtcfg, csr, d_gdata, infront, outfront);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  gettimeofday(&t1, nullptr);
  double dur = (double)(t1.tv_usec - t0.tv_usec) / K / 1000.0;  // ms

  return dur;
}

double RunBaseline1(const RuntimeConfig& rtcfg, const minigun::Csr& csr, GData* gdata) {
  const mg_int N = csr.row_offsets.length - 1;

  // dry run
  custom_kernel::maskedmm_csr_forward_kernel<mg_int, float><<<N, 32>>>(
      csr.row_offsets.data,
      csr.column_indices.data,
      gdata->ndata,
      gdata->ndata,
      gdata->score,
      (int)gdata->D, (int)N, (int)gdata->H);
  CUDA_CALL(cudaDeviceSynchronize());

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    custom_kernel::maskedmm_csr_forward_kernel<mg_int, float><<<N, 32>>>(
        csr.row_offsets.data,
        csr.column_indices.data,
        gdata->ndata,
        gdata->ndata,
        gdata->score,
        (int)gdata->D, (int)N, (int)gdata->H);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  gettimeofday(&t1, nullptr);
  double dur = (double)(t1.tv_usec - t0.tv_usec) / K / 1000.0;  // ms

  return dur;
}

int main(int argc, char** argv) {
  srand(42);
  if (argc < 4) {
    std::cout << "USAGE: ./bench_masked_mm <file_name> <feat_size> <num_heads>" << std::endl;
    return 1;
  }
  const char* filename = argv[1];
  const int feat_size = std::atoi(argv[2]);
  const int num_heads = std::atoi(argv[3]);
  std::cout << "filename=" << filename << " feat_size=" << feat_size
    << " num_heads=" << num_heads << std::endl;

  utils::SampleCsr scsr;
  utils::LoadGraphFromFile(filename, &scsr);
  const mg_int N = scsr.row_offsets.size() - 1;
  const mg_int M = scsr.column_indices.size();
  std::cout << "#Nodes: " << N << " #Edges: " << M << std::endl;

  // csr
  minigun::Csr csr = utils::ToMinigunCsr(scsr, kDLGPU);

  // gdata
  GData gdata;
  gdata.D = feat_size;
  gdata.H = num_heads;
  InitGData(&gdata, N);
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
