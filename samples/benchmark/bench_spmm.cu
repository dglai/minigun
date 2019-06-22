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
                  const minigun::IntCsr& csr,
                  int32_t feat_size) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // gdata
  GData gdata, truth;
  gdata.D = feat_size;
  InitGData(scsr, &gdata, &truth);
  CUDA_CALL(cudaDeviceSynchronize());
 
  // create stream
  RuntimeConfig rtcfg;
  rtcfg.ctx = {kDLGPU, 0};
  int nt = utils::_FindNumThreads(gdata.D, 64);
  rtcfg.data_num_threads = nt;
  rtcfg.data_num_blocks = (gdata.D + (nt * 4) - 1) / (nt * 4);
  CUDA_CALL(cudaStreamCreate(&rtcfg.stream));

  minigun::IntArray infront;

  // check accuracy
  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, int32_t, Config, GData, SPMMFunctor>(
      rtcfg, csr, &gdata, infront);
  CUDA_CALL(cudaDeviceSynchronize());
  CheckResult(scsr, &gdata, &truth);

  // warm up
  const int K = 10;
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLGPU, int32_t, Config, GData, SPMMFunctor>(
        rtcfg, csr, &gdata, infront);
  }
  CUDA_CALL(cudaDeviceSynchronize());

  cudaEventRecord(start);
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLGPU, int32_t, Config, GData, SPMMFunctor>(
        rtcfg, csr, &gdata, infront);
  }
  cudaEventRecord(stop);
  CUDA_CALL(cudaDeviceSynchronize());
  float dur = 0;
  cudaEventElapsedTime(&dur, start, stop);

  FreeGData(&gdata, &truth);

  return dur / K;
}

double RunBaseline1(const utils::SampleCsr& scsr,
                    const minigun::IntCsr& csr,
                    int32_t feat_size) {
  const int32_t N = csr.row_offsets.length - 1;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

   // gdata
  GData gdata, truth;
  gdata.D = feat_size;
  InitGData(scsr, &gdata, &truth);

  int nt = utils::_FindNumThreads(gdata.D, 512);

  const int K = 10;
  // warm up
  for (int i = 0; i < K; ++i) {
    custom_kernel::vector_spmm_forward_kernel_no_eid<int32_t, float><<<N, nt>>>(
        csr.row_offsets.data,
        csr.column_indices.data,
        gdata.weight,
        gdata.ndata,
        gdata.out,
        (int)gdata.D, (int)N);
  }
  CUDA_CALL(cudaDeviceSynchronize());

  cudaEventRecord(start);
  for (int i = 0; i < K; ++i) {
    custom_kernel::vector_spmm_forward_kernel_no_eid<int32_t, float><<<N, nt>>>(
        csr.row_offsets.data,
        csr.column_indices.data,
        gdata.weight,
        gdata.ndata,
        gdata.out,
        (int)gdata.D, (int)N);
  }
  cudaEventRecord(stop);
  CUDA_CALL(cudaDeviceSynchronize());
  float dur = 0;
  cudaEventElapsedTime(&dur, start, stop);

  FreeGData(&gdata, &truth);

  return dur / K;
}

int main(int argc, char** argv) {
  srand(42);
  if (argc < 3) {
    std::cout << "USAGE: ./bench_masked_mm <file_name> <feat_size>" << std::endl;
    return 1;
  }
  const char* filename = argv[1];
  const int feat_size = std::atoi(argv[2]);
  //std::cout << "filename=" << filename << " feat_size=" << feat_size

  utils::SampleCsr scsr;
  utils::LoadGraphFromFile(filename, &scsr);
  const int32_t N = scsr.row_offsets.size() - 1;
  const int32_t M = scsr.column_indices.size();
  //std::cout << "#Nodes: " << N << " #Edges: " << M << std::endl;

  // csr
  minigun::IntCsr csr = utils::ToMinigunCsr(scsr, kDLGPU);

  double dur1 = RunBaseline1(scsr, csr, feat_size);
  double dur2 = RunMinigun(scsr, csr, feat_size);
  std::cout << N << "," << M << "," << feat_size << "," << dur1 << "," << dur2 << "\n";

  return 0;
}
