#include <iostream>
#include <cstdlib>
#include <limits>
#include <time.h>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "./baseline/yzh_kernels.cuh"
#include "./minigun/esoftmax_back.cuh"
#include "../samples_io.h"
#include "../samples_utils.h"

using minigun::advance::RuntimeConfig;
using namespace esoftmax_back;

double RunMinigun(const utils::SampleCsr& scsr,
                  const minigun::IntCsr& csr,
                  int32_t feat_size, int32_t num_heads) {
  // gdata
  GData gdata, truth;
  gdata.H = num_heads;
  InitGData(scsr, &gdata, &truth);
  CUDA_CALL(cudaDeviceSynchronize());
 
  // create stream
  RuntimeConfig rtcfg;
  rtcfg.ctx = {kDLGPU, 0};
  int nt = utils::_FindNumThreads(gdata.H, 32);
  rtcfg.data_num_threads = nt;
  rtcfg.data_num_blocks = (gdata.H + nt - 1) / nt;
  CUDA_CALL(cudaStreamCreate(&rtcfg.stream));

  minigun::IntArray infront;

  // dry run
  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, int32_t, Config, GData, BackSoftmaxAccum>(
      rtcfg, csr, &gdata, infront);
  minigun::advance::Advance<kDLGPU, int32_t, Config, GData, BackSoftmaxMinus>(
      rtcfg, csr, &gdata, infront);
  CUDA_CALL(cudaDeviceSynchronize());
  CheckResult(scsr, &gdata, &truth);

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLGPU, int32_t, Config, GData, BackSoftmaxAccum>(
        rtcfg, csr, &gdata, infront);
    minigun::advance::Advance<kDLGPU, int32_t, Config, GData, BackSoftmaxMinus>(
        rtcfg, csr, &gdata, infront);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  gettimeofday(&t1, nullptr);
  double dur = (double)(t1.tv_sec * 1e6 + t1.tv_usec -
      (t0.tv_sec * 1e6 + t0.tv_usec)) / K / 1000.0;  // ms

  FreeGData(&gdata, &truth);

  return dur;
}

std::pair<utils::SampleCsr, std::vector<int32_t>> Transpose(const utils::SampleCsr& csr) {
  const int32_t N = csr.row_offsets.size() - 1;
  const int32_t M = csr.column_indices.size();
  std::vector<std::vector<int32_t>> adjlist(N);
  std::vector<std::vector<int32_t>> adjlist_e(N);
  for (int32_t u = 0; u < csr.row_offsets.size() - 1; ++u) {
    for (int32_t e = csr.row_offsets[u]; e < csr.row_offsets[u+1]; ++e) {
      int32_t v = csr.column_indices[e];
      adjlist[v].push_back(u);
      adjlist_e[v].push_back(e);
    }
  }
  utils::SampleCsr ret;
  ret.row_offsets.resize(N+1);
  ret.column_indices.resize(M);
  ret.row_offsets[0] = 0;
  std::vector<int32_t> eid(M);
  for (int32_t u = 0; u < N; ++u) {
    ret.row_offsets[u+1] = ret.row_offsets[u] + adjlist[u].size();
    std::copy(adjlist[u].begin(), adjlist[u].end(),
              ret.column_indices.begin() + ret.row_offsets[u]);
    std::copy(adjlist_e[u].begin(), adjlist_e[u].end(),
              eid.begin() + ret.row_offsets[u]);
  }
  return std::make_pair(ret, eid);
}

double RunBaseline1(const utils::SampleCsr& scsr,
                    const minigun::IntCsr& csr,
                    int32_t feat_size, int32_t num_heads) {
  const int32_t N = csr.row_offsets.length - 1;
  const int32_t M = csr.column_indices.length;
  const int H = num_heads;
  if (H > 64) {
    // thread block can have at most 64 threads in z dimension.
    return 100.0;
  }

  // gdata
  GData gdata, truth;
  gdata.H = num_heads;
  InitGData(scsr, &gdata, &truth);

  const auto& trans = Transpose(scsr);
  const auto& csr_t = utils::ToMinigunCsr(trans.first, kDLGPU);
  int32_t* d_eid;
  CUDA_CALL(cudaMalloc(&d_eid, sizeof(int32_t) * M));
  CUDA_CALL(cudaMemcpy(d_eid, &trans.second[0], sizeof(int32_t) * M,
        cudaMemcpyHostToDevice));

  // dry run
  custom_kernel::sparse_softmax_backward_kernel<int32_t, float>
    <<<dim3(N, 1, 1), dim3(1, 256/H, H)>>>(
      csr_t.row_offsets.data,
      d_eid,
      gdata.grad_score,
      gdata.score,
      gdata.out,
      (int)N, (int)H);
  CUDA_CALL(cudaDeviceSynchronize());
  CheckResult(scsr, &gdata, &truth);

  const int K = 10;
  timeval t0, t1;
  gettimeofday(&t0, nullptr);
  for (int i = 0; i < K; ++i) {
    custom_kernel::sparse_softmax_backward_kernel<int32_t, float>
      <<<dim3(N, 1, 1), dim3(1, 256/H, H)>>>(
        csr_t.row_offsets.data,
        d_eid,
        gdata.grad_score,
        gdata.score,
        gdata.out,
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
    std::cout << "USAGE: ./bench_backward_edge_softmax <file_name> <num_heads>" << std::endl;
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

  double dur1 = RunMinigun(scsr, csr, 0, num_heads);
  std::cout << "minigun time(ms): " << dur1 << std::endl;
  double dur2 = RunBaseline1(scsr, csr, 0, num_heads);
  std::cout << "baseline1 time(ms): " << dur2 << std::endl;


  return 0;
}
