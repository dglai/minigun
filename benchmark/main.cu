/* Benchmark code for all edge policy. */
#include <iostream>
#include <cstdlib>
#include <limits>
#include <ctime>
#include <cuda_runtime.h>

#include <minigun/minigun.h>
#include "benchmark_utils.h"
#include "minigun/graph_ops.cuh"
#include "custom_kernel/graph_ops.cuh"
#include "ground_truth/graph_ops.h"

struct GData {
  mg_int head = 0;
  mg_int dim = 0;
  float* ndata{nullptr}; // ndata
  float* sum{nullptr};
  float* max{nullptr};
  float* score{nullptr}; // edata
};

const mg_int D = 64;  // dimension of each head. 
const mg_int H = 8;   // number of heads.


int main(int argc, char **argv) {
  srand(42);

  // create graph
  std::vector<mg_int> row_offsets, column_indices;
  std::vector<mg_int> row_offsets_t, column_indices_t;
  utils::CreateSparseBatch1(128, row_offsets, column_indices);
  //utils::CreateDatasetGraph("data/citeseer.txt", row_offsets, column_indices); 
  utils::CsrTranspose(row_offsets, column_indices, row_offsets_t, column_indices_t);
    const mg_int N = row_offsets.size() - 1;
  const mg_int M = column_indices.size();
  std::cout << "#nodes: " << N << " #edges: " << M << " #feats: " << H << "x" << D << std::endl; 
  
  // copy graph to gpu
  CUDA_CALL(cudaSetDevice(0));
  minigun::Csr csr, csr_t;
  minigun::IntArray1D infront, outfront;
  csr.row_offsets.length = row_offsets.size();
  csr_t.row_offsets.length = row_offsets_t.size();
  CUDA_CALL(cudaMalloc(&csr.row_offsets.data, sizeof(mg_int) * row_offsets.size()));
  CUDA_CALL(cudaMemcpy(csr.row_offsets.data, &row_offsets[0],
        sizeof(mg_int) * row_offsets.size(), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&csr_t.row_offsets.data, sizeof(mg_int) * row_offsets_t.size()));
  CUDA_CALL(cudaMemcpy(csr_t.row_offsets.data, &row_offsets_t[0],
        sizeof(mg_int) * row_offsets_t.size(), cudaMemcpyHostToDevice));
  csr.column_indices.length = column_indices.size();
  csr_t.column_indices.length = column_indices_t.size();
  CUDA_CALL(cudaMalloc(&csr.column_indices.data, sizeof(mg_int) * column_indices.size()));
  CUDA_CALL(cudaMemcpy(csr.column_indices.data, &column_indices[0],
        sizeof(mg_int) * column_indices.size(), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&csr_t.column_indices.data, sizeof(mg_int) * column_indices_t.size()));
  CUDA_CALL(cudaMemcpy(csr_t.column_indices.data, &column_indices_t[0],
        sizeof(mg_int) * column_indices_t.size(), cudaMemcpyHostToDevice));
 
  // Create feature data
  std::vector<float> vvec(N * H * D), evec(M * H), vmax(N * H * D);
  for (mg_int i = 0; i < N * H * D; ++i) {
    vvec[i] = (float)rand() / RAND_MAX;
    vmax[i] = std::numeric_limits<float>::lowest();
  }
  for (mg_int i = 0; i < M * H; ++i) {
    evec[i] = 0.;
  }

  // Copy feature data to gpu
  GData gdata;
  gdata.dim = D;
  gdata.head = H;
  CUDA_CALL(cudaMalloc(&gdata.ndata, sizeof(float) * N * H * D));
  CUDA_CALL(cudaMemcpy(gdata.ndata, &vvec[0], sizeof(float) * N * H * D, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&gdata.sum, sizeof(float) * N * H));
  CUDA_CALL(cudaMemset(gdata.sum, 0, sizeof(float) * N * H));
  CUDA_CALL(cudaMalloc(&gdata.max, sizeof(float) * N * H));
  CUDA_CALL(cudaMemcpy(gdata.max, &vmax[0], sizeof(float) * N * H, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&gdata.score, sizeof(float) * M * H));
  CUDA_CALL(cudaMemcpy(gdata.score, &evec[0], sizeof(float) * M * H, cudaMemcpyHostToDevice));
  GData* d_gdata;
  CUDA_CALL(cudaMalloc(&d_gdata, sizeof(GData)));
  CUDA_CALL(cudaMemcpy(d_gdata, &gdata, sizeof(GData), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaDeviceSynchronize());

  /*
   * Start of Masked GEMM
   */

  // Create stream
  minigun::advance::RuntimeConfig config_masked_mm;
  int nt = utils::_FindNumThreads(D * H, 32);
  config_masked_mm.data_num_threads = nt;
  config_masked_mm.data_num_blocks = 1;
  CUDA_CALL(cudaStreamCreate(&config_masked_mm.stream));
 
  // Compute ground truth
  clock_t tic = clock();
  std::vector<float> truth_0 = ground_truth::MaskedMM(H, D, row_offsets, column_indices, vvec);
  std::cout << "ground-truth-maskedmm: " << double(clock() - tic) / CLOCKS_PER_SEC << " s" << std::endl;

  // Custom kernel
  float *rst_arr;
  CUDA_CALL(cudaMalloc(&rst_arr, sizeof(float) * M * H));
  CUDA_CALL(cudaMemcpy(rst_arr, &evec[0], sizeof(float) * M * H, cudaMemcpyHostToDevice));
  tic = clock();
  custom_kernel::maskedmm_csr_forward_kernel<mg_int, float><<<N, 32>>>(csr.row_offsets.data, csr.column_indices.data, gdata.ndata, gdata.ndata, rst_arr, (int)D, (int)N, (int)H);
  CUDA_CALL(cudaDeviceSynchronize());
  std::cout << "custom-kernel-maskedmm: " << double(clock() - tic) / CLOCKS_PER_SEC << " s" << std::endl;

  // Advance
  tic = clock();
  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLGPU, Config, GData, functor::MaskedMM<GData>>(
      config_masked_mm, csr, d_gdata, infront, outfront);

  CUDA_CALL(cudaDeviceSynchronize());
  std::cout << "minigun-maskedmm: " << double(clock() - tic) / CLOCKS_PER_SEC << " s" << std::endl;

  // Verify result
  std::vector<float> rst_0(M * H);
  CUDA_CALL(cudaMemcpy(&rst_0[0], gdata.score, sizeof(float) * M * H, cudaMemcpyDeviceToHost));
  std::cout << "Correct? " << utils::VecAllClose(truth_0, rst_0) << std::endl;
  CUDA_CALL(cudaMemcpy(&rst_0[0], rst_arr, sizeof(float) * M * H, cudaMemcpyDeviceToHost));
  std::cout << "Correct? " << utils::VecAllClose(truth_0, rst_0) << std::endl;

  /*
   * Start of sparse softmax.
   */

  // Create stream
  minigun::advance::RuntimeConfig config_softmax;
  nt = utils::_FindNumThreads(H, 32);
  config_softmax.data_num_threads = nt;
  config_softmax.data_num_blocks = (H + nt - 1) / nt;
  CUDA_CALL(cudaStreamCreate(&config_softmax.stream));

  // Compute ground truth
  tic = clock();
  std::vector<float> truth_1 = ground_truth::Softmax(H, row_offsets, column_indices, truth_0);
  std::cout << "ground-truth-softmax: " << double(clock() - tic) / CLOCKS_PER_SEC << " s" << std::endl;

  // Custom kernel
  float *rst_arr_1;
  CUDA_CALL(cudaMalloc(&rst_arr_1, sizeof(float) * M * H));
  CUDA_CALL(cudaMemcpy(rst_arr, &evec[0], sizeof(float) * M * H, cudaMemcpyHostToDevice));
  tic = clock();
  custom_kernel::sparse_softmax_forward_kernel<mg_int, float><<<(N + 31) / 32, dim3(32, H)>>>(csr_t.row_offsets.data, gdata.score, rst_arr_1, (int)N, (int)H);
  CUDA_CALL(cudaDeviceSynchronize());
  std::cout << "custom-kernel-softmax: " << double(clock() - tic) / CLOCKS_PER_SEC << " s" << std::endl;

  // Use the same Config policy: advance_all, kV2N
  tic = clock();
  minigun::advance::Advance<kDLGPU, Config, GData, functor::EdgeMax<GData>>(
      config_softmax, csr, d_gdata, infront, outfront);
  minigun::advance::Advance<kDLGPU, Config, GData, functor::MinuxMaxExpSum<GData>>(
      config_softmax, csr, d_gdata, infront, outfront);
  minigun::advance::Advance<kDLGPU, Config, GData, functor::Norm<GData>>(
      config_softmax, csr, d_gdata, infront, outfront);
  CUDA_CALL(cudaDeviceSynchronize());
  std::cout << "minigun-softmax: " << double(clock() - tic) / CLOCKS_PER_SEC << " s" << std::endl;


  // output
  std::vector<float> rst_1(M * H);
  CUDA_CALL(cudaMemcpy(&rst_1[0], gdata.score, sizeof(float) * M * H, cudaMemcpyDeviceToHost));
  std::cout << "Correct? " << utils::VecAllClose(truth_1, rst_1) << std::endl;
  //CUDA_CALL(cudaMemcpy(&rst_1[0], rst_arr_1, sizeof(float) * M * H, cudaMemcpyDeviceToHost));
  //std::cout << "Correct? " << utils::VecAllClose(truth_1, rst_1) << std::endl;

  return 0;
}
