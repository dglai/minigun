#include <iostream>
#include <cstdlib>
#include <limits>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "./baseline/yzh_kernels.cuh"
#include "./minigun/esoftmax.cuh"
#include "../samples_io.h"
#include "../samples_utils.h"

using minigun::advance::RuntimeConfig;
using namespace esoftmax;

double RunMinigun(const utils::SampleCsr& scsr,
                  const minigun::IntSpMat& spmat,
                  const minigun::IntArray& eid_mapping,
                  int32_t feat_size, int32_t num_heads) {
  // gdata
  GData gdata, truth;
  gdata.H = num_heads;
  InitGData(scsr, eid_mapping, &gdata, &truth);
  CUDA_CALL(cudaDeviceSynchronize());
 
  // create stream
  RuntimeConfig rtcfg;
  rtcfg.ctx = {kDLGPU, 0};
  int nt = utils::_FindNumThreads(gdata.H, 32);
  rtcfg.data_num_threads = nt;
  rtcfg.data_num_blocks = gdata.H / nt;
  CUDA_CALL(cudaStreamCreate(&rtcfg.stream));

  // dry run
  typedef minigun::advance::Config<minigun::advance::kDst> Config;
  minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, EdgeMax>(
      rtcfg, spmat, &gdata);
  minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, MinusMaxExpSum>(
      rtcfg, spmat, &gdata);
  minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, NormByDst>(
      rtcfg, spmat, &gdata);
  CUDA_CALL(cudaDeviceSynchronize());
  CheckResult(scsr, &gdata, &truth);

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, EdgeMax>(
        rtcfg, spmat, &gdata);
    minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, MinusMaxExpSum>(
        rtcfg, spmat, &gdata);
    minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, NormByDst>(
        rtcfg, spmat, &gdata);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  gettimeofday(&t1, nullptr);
  double dur = (double)(t1.tv_sec * 1e6 + t1.tv_usec -
      (t0.tv_sec * 1e6 + t0.tv_usec)) / K / 1000.0;  // ms

  FreeGData(&gdata, &truth);

  return dur;
}

double RunBaseline1(const utils::SampleCsr& scsr,
                  const minigun::IntCsr& csr,
                  const minigun::IntArray& eid_mapping,
                  int32_t feat_size, int32_t num_heads) {
  // gdata
  GData gdata, truth;
  gdata.H = num_heads;
  InitGData(scsr, eid_mapping, &gdata, &truth);
  CUDA_CALL(cudaDeviceSynchronize());
 
  const int32_t N = csr.row_offsets.length - 1;
  const int H = gdata.H;

  // dry run
  custom_kernel::sparse_softmax_forward_kernel<int32_t, float><<<(N + 31) / 32, dim3(32, H)>>>(
      csr.row_offsets.data,
      gdata.score,
      gdata.ret,
      (int)N, (int)H);
  CUDA_CALL(cudaDeviceSynchronize());

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    custom_kernel::sparse_softmax_forward_kernel<int32_t, float><<<(N + 31) / 32, dim3(32, H)>>>(
        csr.row_offsets.data,
        gdata.score,
        gdata.ret,
        (int)N, (int)H);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  gettimeofday(&t1, nullptr);
  double dur = (double)(t1.tv_sec * 1e6 + t1.tv_usec -
      (t0.tv_sec * 1e6 + t0.tv_usec)) / K / 1000.0;  // ms

  FreeGData(&gdata, &truth);

  return dur;
}

int main(int argc, char** argv) {
  srand(42);
  if (argc < 3) {
    std::cout << "USAGE: ./bench_edge_softmax <file_name> <num_heads>" << std::endl;
    return 1;
  }
  const char* filename = argv[1];
  const int num_heads = std::atoi(argv[2]);
  std::cout << "filename=" << filename << " num_heads=" << num_heads << std::endl;

  utils::SampleCsr scsr;
  utils::LoadGraphFromFile(filename, &scsr);
  const int32_t N = scsr.row_offsets.size() - 1;
  const int32_t M = scsr.column_indices.size();
  std::cout << "#Nodes: " << N << " #Edges: " << M << std::endl;

  // csr
  minigun::IntCsr csr = utils::ToMinigunCsr(scsr, kDLGPU);
  auto csr_mapping = utils::arange(0, M, kDLGPU);
  auto pack = utils::ToMinigunReverseCsr(scsr, csr_mapping, kDLGPU);
  minigun::IntCsr csr_t = pack.first;
  minigun::IntArray csr_t_mapping = pack.second;
  minigun::IntSpMat spmat = {nullptr, &csr_t, nullptr};

  double dur1 = RunMinigun(scsr, spmat, csr_t_mapping, 0, num_heads);
  std::cout << "minigun time(ms): " << dur1 << std::endl;
  double dur2 = RunBaseline1(scsr, csr, csr_t_mapping, 0, num_heads);
  std::cout << "baseline1 time(ms): " << dur2 << std::endl;

  return 0;
}
