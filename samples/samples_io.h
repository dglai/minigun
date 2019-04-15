#ifndef SAMPLES_SAMPLES_IO_H_
#define SAMPLES_SAMPLES_IO_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>
#include "./samples_utils.h"

namespace utils {

// Csr graph that owns memory; easier for prototyping
struct SampleCsr {
  std::vector<mg_int> row_offsets;
  std::vector<mg_int> column_indices;
};

}  // namespace utils

namespace dmlc {
namespace serializer {

template <>
struct Handler<utils::SampleCsr> {
  inline static void Write(Stream* strm, const utils::SampleCsr& csr) {
    Handler<std::vector<mg_int>>::Write(strm, csr.row_offsets);
    Handler<std::vector<mg_int>>::Write(strm, csr.column_indices);
  }
  inline static bool Read(Stream* strm, utils::SampleCsr* out_csr) {
    if (!Handler<std::vector<mg_int>>::Read(strm, &(out_csr->row_offsets))) return false;
    if (!Handler<std::vector<mg_int>>::Read(strm, &(out_csr->column_indices))) return false;
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

// create a minigun Csr that copies the given sample csr memory
__inline__ minigun::Csr ToMinigunCsr(const SampleCsr& sample_csr, DLDeviceType device) {
  minigun::Csr csr;
  const size_t rsize = sample_csr.row_offsets.size();
  const size_t csize = sample_csr.column_indices.size();
  if (device == kDLCPU) {
    csr.row_offsets.length = rsize;
    csr.row_offsets.data = new mg_int[rsize];
    std::copy(sample_csr.row_offsets.begin(), sample_csr.row_offsets.end(),
        csr.row_offsets.data);
    csr.column_indices.length = csize;
    csr.column_indices.data = new mg_int[csize];
    std::copy(sample_csr.column_indices.begin(), sample_csr.column_indices.end(),
        csr.column_indices.data);
#ifdef __CUDACC__
  } else if (device == kDLGPU) {
    csr.row_offsets.length = rsize;
    CUDA_CALL(cudaMalloc(&csr.row_offsets.data, rsize * sizeof(mg_int)));
    CUDA_CALL(cudaMemcpy(csr.row_offsets.data, &sample_csr.row_offsets[0],
          sizeof(mg_int) * rsize, cudaMemcpyHostToDevice));
    csr.column_indices.length = csize;
    CUDA_CALL(cudaMalloc(&csr.column_indices.data, csize * sizeof(mg_int)));
    CUDA_CALL(cudaMemcpy(csr.column_indices.data, &sample_csr.column_indices[0],
          sizeof(mg_int) * csize, cudaMemcpyHostToDevice));
#endif  // __CUDACC__
  } else {
    LOG(INFO) << "Unsupported device: " << device;
  }
  return csr;
}

// create a sample csr that COPIES the memory of the minigun csr
__inline__ SampleCsr ToSampleCsr(const minigun::Csr& mg_csr) {
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
