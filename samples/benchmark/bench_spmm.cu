#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "./baseline/yzh_kernels.cuh"
#include "./minigun/spmm.cuh"
#include "../samples_io.h"
#include "../samples_utils.h"

using minigun::advance::RuntimeConfig;
using namespace spmm;

double RunMinigun(const utils::SampleCsr& scsr,
                  const minigun::Csr& csr,
                  mg_int feat_size, mg_int num_heads) {
  // gdata
  GData gdata, truth;
  gdata.D = feat_size;
  gdata.H = num_heads;
  InitGData(scsr, &gdata, &truth);
  GData* d_gdata;
  CUDA_CALL(cudaMalloc(&d_gdata, sizeof(GData)));
  CUDA_CALL(cudaMemcpy(d_gdata, &gdata, sizeof(GData), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaDeviceSynchronize());
 
  // create stream
  RuntimeConfig rtcfg;
  rtcfg.ctx = {kDLGPU, 0};
  int nt = utils::_FindNumThreads(gdata.H * gdata.D, 64);
  rtcfg.data_num_threads = nt;
  rtcfg.data_num_blocks = (gdata.H * gdata.D + (nt * 4) - 1) / (nt * 4);
  CUDA_CALL(cudaStreamCreate(&rtcfg.stream));

  minigun::IntArray1D infront, outfront;

  // dry run
  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, Config, GData, SPMMFunctor>(
      rtcfg, csr, d_gdata, infront, outfront);
  CUDA_CALL(cudaDeviceSynchronize());
  CheckResult(scsr, &gdata, &truth);

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLGPU, Config, GData, SPMMFunctor>(
        rtcfg, csr, d_gdata, infront, outfront);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  gettimeofday(&t1, nullptr);
  double dur = (double)(t1.tv_sec * 1e6 + t1.tv_usec -
      (t0.tv_sec * 1e6 + t0.tv_usec)) / K / 1000.0;  // ms

  FreeGData(&gdata, &truth);

  return dur;
}

double RunBaseline1(const utils::SampleCsr& scsr,
                    const minigun::Csr& csr,
                    mg_int feat_size, mg_int num_heads) {
  const mg_int N = csr.row_offsets.length - 1;

   // gdata
  GData gdata, truth;
  gdata.D = feat_size;
  gdata.H = num_heads;
  InitGData(scsr, &gdata, &truth);

  // dry run
  custom_kernel::vector_spmm_forward_kernel_no_eid<mg_int, float><<<N, 32>>>(
      csr.row_offsets.data,
      csr.column_indices.data,
      gdata.weight,
      gdata.ndata,
      gdata.out,
      (int)gdata.D, (int)N, (int)gdata.H);
  CUDA_CALL(cudaDeviceSynchronize());

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    custom_kernel::vector_spmm_forward_kernel_no_eid<mg_int, float><<<N, 32>>>(
        csr.row_offsets.data,
        csr.column_indices.data,
        gdata.weight,
        gdata.ndata,
        gdata.out,
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
  const mg_int N = scsr.row_offsets.size() - 1;
  const mg_int M = scsr.column_indices.size();
  std::cout << "#Nodes: " << N << " #Edges: " << M << std::endl;

  // csr
  minigun::Csr csr = utils::ToMinigunCsr(scsr, kDLGPU);

  double dur1 = RunMinigun(scsr, csr, feat_size, num_heads);
  std::cout << "minigun time(ms): " << dur1 << std::endl;
  double dur2 = RunBaseline1(scsr, csr, feat_size, num_heads);
  std::cout << "baseline1 time(ms): " << dur2 << std::endl;

  return 0;
}
