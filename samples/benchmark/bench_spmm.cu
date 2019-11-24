#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <minigun/minigun.h>
#include "./baseline/yzh_kernels.cuh"
#include "./minigun/spmm.cuh"
#include "../samples_io.h"
#include "../samples_utils.h"

using minigun::advance::RuntimeConfig;
using namespace spmm;

double RunMinigun(const utils::SampleCsr& scsr,
                  const minigun::IntSpMat& spmat,
                  int32_t feat_size,
                  GData& gdata,
                  GData& truth) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // create stream
  RuntimeConfig rtcfg;
  rtcfg.ctx = {kDLGPU, 0};
  int nt = utils::_FindNumThreads(gdata.D, 512);
  rtcfg.data_num_threads = nt;
  rtcfg.data_num_blocks = (gdata.D + (nt * 4) - 1) / (nt * 4);
  CUDA_CALL(cudaStreamCreate(&rtcfg.stream));

  minigun::IntArray infront;
  ResetGData(&gdata, scsr.row_offsets.size() - 1);

  // check accuracy
  typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kDst> Config;
  minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, SPMMFunctor>(
      rtcfg, spmat, &gdata, infront);
  CUDA_CALL(cudaDeviceSynchronize());
  CheckResult(scsr, &gdata, &truth);

  // warm up
  const int K = 10;
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, SPMMFunctor>(
        rtcfg, spmat, &gdata, infront);
  }

  cudaEventRecord(start);
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLGPU, int32_t, float, Config, GData, SPMMFunctor>(
        rtcfg, spmat, &gdata, infront);
  }
  cudaEventRecord(stop);
  CUDA_CALL(cudaEventSynchronize(stop));
  float dur = 0;
  cudaEventElapsedTime(&dur, start, stop);

  return dur / K;
}

double RunBaseline1(const utils::SampleCsr& scsr,
                    const minigun::IntCsr& csr,
                    int32_t feat_size,
                    GData& gdata,
                    GData& truth) {
  const int32_t N = csr.row_offsets.length - 1;
  ResetGData(&gdata, N);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int nt = utils::_FindNumThreads(gdata.D, 512);

  custom_kernel::vector_spmm_forward_kernel_no_eid<int32_t, float><<<N, nt>>>(
      csr.row_offsets.data,
      csr.column_indices.data,
      gdata.weight,
      gdata.ndata,
      gdata.out,
      (int)gdata.D, (int)N);
  CUDA_CALL(cudaDeviceSynchronize());
  CheckResult(scsr, &gdata, &truth);

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
  CUDA_CALL(cudaEventSynchronize(stop));
  float dur = 0;
  cudaEventElapsedTime(&dur, start, stop);


  return dur / K;
}

double RunBaseline2(const utils::SampleCsr& scsr,
                    const minigun::IntCsr& csr,
                    int32_t feat_size,
                    GData& gdata,
                    GData& truth) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int n = csr.row_offsets.length - 1;
  ResetGData(&gdata, n);
  int k = feat_size;
  int nnz = scsr.row_offsets[n];

  float alpha = 1.0;
  float beta = 0.0;

  cublasStatus_t stat;
  cublasHandle_t cublas_handle{nullptr};
  stat = cublasCreate(&cublas_handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
  }

  cusparseStatus_t status;
  cusparseHandle_t handle=0;
  cusparseMatDescr_t descr=0;

  /* initialize cusparse library */
  status= cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("CUSPARSE Library initialization failed\n");
    exit(-1);
  }

  /* create and setup matrix descriptor */
  status= cusparseCreateMatDescr(&descr);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix descriptor initialization failed\n");
    exit(-1);
  }
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

  status= cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
          n, k, n, nnz, &alpha, descr, gdata.weight, csr.row_offsets.data, csr.column_indices.data,
          gdata.ndata, k, &beta, gdata.out, n);
  // transpose results to check correctness
  CUDA_CALL(cudaDeviceSynchronize());
  float* t;
  CUDA_CALL(cudaMalloc(&t, sizeof(float) * n * k));
  stat = cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k, n, &alpha, gdata.out, n, &beta, NULL, k, t, k);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS transpose failed\n");
    return EXIT_FAILURE;
  }
  GData gdata2;
  gdata2.out = t;
  CheckResult(scsr, &gdata2, &truth);
  CUDA_CALL(cudaFree(t));
  CUDA_CALL(cudaDeviceSynchronize());

  const int K = 10;

  // warm up
  for (int i = 0; i < K; ++i) {
    status= cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
            n, k, n, nnz, &alpha, descr, gdata.weight, csr.row_offsets.data, csr.column_indices.data,
            gdata.ndata, k, &beta, gdata.out, n);
  }
  cudaEventRecord(start);
  for (int i = 0; i < K; ++i) {
    status= cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
            n, k, n, nnz, &alpha, descr, gdata.weight, csr.row_offsets.data, csr.column_indices.data,
            gdata.ndata, k, &beta, gdata.out, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float dur = 0;
  cudaEventElapsedTime(&dur, start, stop);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix-vector multiplication failed\n");
    exit(-1);
  }

  /* destroy matrix descriptor */
  status = cusparseDestroyMatDescr(descr);
  descr = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix descriptor destruction failed\n");
    exit(-1);
  }

  /* destroy handle */
  status = cusparseDestroy(handle);
  handle = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("CUSPARSE Library release of resources failed\n");
    exit(-1);
  }
  cusparseDestroyMatDescr(descr);
  cusparseDestroy(handle);
  cublasDestroy(cublas_handle);
  return dur / K;
}

int main(int argc, char** argv) {
  // test transpose
  srand(42);
  /*
  // Small testing graph
  const int32_t a[] = {0, 2, 5, 5, 7};
  const int32_t b[] = {1, 2, 0, 2, 3, 0, 1};
  auto scsr = utils::SampleCsr{std::vector<int32_t>(std::begin(a), std::end(a)), std::vector<int32_t>(std::begin(b), std::end(b))};
  const int feat_size = 2;
  */
  if (argc < 3) {
    std::cout << "USAGE: ./bench_spmm <file_name> <feat_size>" << std::endl;
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
  minigun::IntCoo coo = utils::ToMinigunCoo(scsr, kDLGPU);
  minigun::IntCsr csr = utils::ToMinigunCsr(scsr, kDLGPU);
  auto csr_mapping = utils::arange(0, M, kDLGPU);
  auto pack = utils::ToMinigunReverseCsr(scsr, csr_mapping, kDLGPU);
  minigun::IntCsr csr_t = pack.first;
  minigun::IntArray csr_t_mapping = pack.second;
  minigun::IntSpMat spmat = {&csr, &csr_t, &coo};

  // gdata
  GData gdata, truth;
  gdata.D = feat_size;
  InitGData(scsr, csr_t_mapping, &gdata, &truth);
  CUDA_CALL(cudaDeviceSynchronize());

  //double dur1 = 0;
  double dur1 = RunBaseline1(scsr, csr_t, feat_size, gdata, truth);
  //double dur2 = 0;
  double dur2 = RunBaseline2(scsr, csr_t, feat_size, gdata, truth);
  //double dur3 = 0;
  double dur3 = RunMinigun(scsr, spmat, feat_size, gdata, truth);
  std::cout << N << "," << M << "," << feat_size << "," << dur1 << "," << dur2 << "," << dur3 << "\n";
  FreeGData(&gdata, &truth);
  cudaDeviceReset();
  return 0;
}
