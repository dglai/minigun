#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "./baseline/yzh_kernels.cuh"
#include "./minigun/masked_mm.cuh"
#include "../samples_io.h"
#include "../samples_utils.h"

using minigun::advance::RuntimeConfig;
using namespace masked_mm;

double RunMinigun(const utils::SampleCsr& scsr,
                  const minigun::IntSpMat& spmat,
                  int32_t feat_size,
                  int32_t num_heads) {
  // gdata
  GData gdata, truth;
  gdata.D = feat_size;
  gdata.H = num_heads;
  InitGData(scsr, &gdata, &truth);
  CUDA_CALL(cudaDeviceSynchronize());
  
  // create stream
  RuntimeConfig cfg;
  cfg.ctx = {kDLGPU, 0};
  int nt = utils::_FindNumThreads(gdata.H, 32);
  cfg.data_num_threads = nt;
  cfg.data_num_blocks = gdata.H / nt;
  CUDA_CALL(cudaStreamCreate(&cfg.stream));

  minigun::IntArray infront;

  // dry run
  typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kEdge> Config;
  minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, MaskedMMFunctor>(
      cfg, csr, &gdata, infront);
  CUDA_CALL(cudaDeviceSynchronize());
  CheckResult(scsr, &gdata, &truth);

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kEdge> Config;
    minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, MaskedMMFunctor>(
        cfg, csr, &gdata, infront);
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
                    int32_t feat_size, int32_t num_heads) {
  const int32_t N = csr.row_offsets.length - 1;

  // gdata
  GData gdata, truth;
  gdata.D = feat_size;
  gdata.H = num_heads;
  InitGData(scsr, &gdata, &truth);

  // dry run
  custom_kernel::maskedmm_csr_forward_kernel<int32_t, float><<<N, 32>>>(
      csr.row_offsets.data,
      csr.column_indices.data,
      gdata.ndata,
      gdata.ndata,
      gdata.score,
      (int)gdata.D, (int)N, (int)gdata.H);
  CUDA_CALL(cudaDeviceSynchronize());

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    custom_kernel::maskedmm_csr_forward_kernel<int32_t, float><<<N, 32>>>(
        csr.row_offsets.data,
        csr.column_indices.data,
        gdata.ndata,
        gdata.ndata,
        gdata.score,
        (int)gdata.D, (int)N, (int)gdata.H);
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
  const int32_t N = scsr.row_offsets.size() - 1;
  const int32_t M = scsr.column_indices.size();
  std::cout << "#Nodes: " << N << " #Edges: " << M << std::endl;

  // csr
  minigun::IntCoo coo = utils::ToMinigunCoo(scsr, kDLGPU);
  minigun::IntSpMat spmat = {nullptr, nullptr, &coo};

  double dur1 = RunMinigun(scsr, spmat, feat_size, num_heads);
  std::cout << "minigun time(ms): " << dur1 << std::endl;
  double dur2 = RunBaseline1(scsr, csr, feat_size, num_heads);
  std::cout << "baseline1 time(ms): " << dur2 << std::endl;

  return 0;
}
