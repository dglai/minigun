#ifndef SAMPLES_SAMPLES_IO_H_
#define SAMPLES_SAMPLES_IO_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>
#include "./samples_utils.h"

namespace utils {

// Csr graph that owns memory; easier for prototyping
struct SampleCsr {
  std::vector<int32_t> row_offsets;
  std::vector<int32_t> column_indices;
};

}  // namespace utils

namespace dmlc {
namespace serializer {

template <>
struct Handler<utils::SampleCsr> {
  inline static void Write(Stream* strm, const utils::SampleCsr& csr) {
    Handler<std::vector<int32_t>>::Write(strm, csr.row_offsets);
    Handler<std::vector<int32_t>>::Write(strm, csr.column_indices);
  }
  inline static bool Read(Stream* strm, utils::SampleCsr* out_csr) {
    if (!Handler<std::vector<int32_t>>::Read(strm, &(out_csr->row_offsets))) return false;
    if (!Handler<std::vector<int32_t>>::Read(strm, &(out_csr->column_indices))) return false;
    return true;
  }
};

}  // namespace serializer
}  // namsespace dmlc

namespace utils {

__inline__ void SaveGraphToFile(const char* filename, const utils::SampleCsr& csr) {
  dmlc::Stream *fs = dmlc::Stream::Create(filename, "w");
  fs->Write(csr);
  delete fs;
}

__inline__ bool LoadGraphFromFile(const char* filename, utils::SampleCsr* out_csr) {
  dmlc::Stream *fs = dmlc::Stream::Create(filename, "r");
  bool ret = fs->Read(out_csr);
  delete fs;
  return ret;
}

// create a minigun Coo that copies the given sample csr memory
__inline__ minigun::IntCoo ToMinigunCoo(const SampleCsr& sample_csr, DLDeviceType device) {
  minigun::IntCoo coo;
  const size_t n_edge = sample_csr.column_indices.size();
  if (device == kDLCPU) {
    assert(0);
#ifdef __CUDACC__
  } else if (device == kDLGPU) {
    int32_t* row = new int32_t[n_edge];
    for (size_t row_id = 0; row_id < sample_csr.row_offsets.size() - 1; ++row_id) {
      for (size_t edge_id = sample_csr.row_offsets[row_id]; edge_id < sample_csr.row_offsets[row_id + 1]; ++edge_id) {
        row[edge_id] = row_id;
      }
    }
    coo.row.length = n_edge;
    CUDA_CALL(cudaMalloc(&coo.row.data, n_edge* sizeof(int32_t)));
    CUDA_CALL(cudaMemcpy(coo.row.data, row,
          sizeof(int32_t) * n_edge, cudaMemcpyHostToDevice));
    coo.column.length = n_edge;
    CUDA_CALL(cudaMalloc(&coo.column.data, n_edge* sizeof(int32_t)));
    CUDA_CALL(cudaMemcpy(coo.column.data, &sample_csr.column_indices[0],
          sizeof(int32_t) * n_edge, cudaMemcpyHostToDevice));
    delete []row;
#endif  // __CUDACC__
  } else {
    LOG(INFO) << "Unsupported device: " << device;
  }
  return coo;
}
// create a minigun Csr that copies the given sample csr memory
__inline__ minigun::IntCsr ToMinigunCsr(const SampleCsr& sample_csr, DLDeviceType device) {
  minigun::IntCsr csr;
  const size_t rsize = sample_csr.row_offsets.size();
  const size_t csize = sample_csr.column_indices.size();
  if (device == kDLCPU) {
    csr.row_offsets.length = rsize;
    csr.row_offsets.data = new int32_t[rsize];
    std::copy(sample_csr.row_offsets.begin(), sample_csr.row_offsets.end(),
        csr.row_offsets.data);
    csr.column_indices.length = csize;
    csr.column_indices.data = new int32_t[csize];
    std::copy(sample_csr.column_indices.begin(), sample_csr.column_indices.end(),
        csr.column_indices.data);
#ifdef __CUDACC__
  } else if (device == kDLGPU) {
    csr.row_offsets.length = rsize;
    CUDA_CALL(cudaMalloc(&csr.row_offsets.data, rsize * sizeof(int32_t)));
    CUDA_CALL(cudaMemcpy(csr.row_offsets.data, &sample_csr.row_offsets[0],
          sizeof(int32_t) * rsize, cudaMemcpyHostToDevice));
    csr.column_indices.length = csize;
    CUDA_CALL(cudaMalloc(&csr.column_indices.data, csize * sizeof(int32_t)));
    CUDA_CALL(cudaMemcpy(csr.column_indices.data, &sample_csr.column_indices[0],
          sizeof(int32_t) * csize, cudaMemcpyHostToDevice));
#endif  // __CUDACC__
  } else {
    LOG(INFO) << "Unsupported device: " << device;
  }
  return csr;
}

// create a sample csr that COPIES the memory of the minigun csr
__inline__ SampleCsr ToSampleCsr(const minigun::IntCsr& mg_csr) {
  SampleCsr csr;
  csr.row_offsets.resize(mg_csr.row_offsets.length);
  std::copy(mg_csr.row_offsets.data, mg_csr.row_offsets.data + mg_csr.row_offsets.length,
            csr.row_offsets.begin());
  csr.column_indices.resize(mg_csr.column_indices.length);
  std::copy(mg_csr.column_indices.data, mg_csr.column_indices.data + mg_csr.column_indices.length,
            csr.column_indices.begin());
  return csr;
}

}  // namespace utils

#endif  // SAMPLES_SAMPLES_IO_H_
