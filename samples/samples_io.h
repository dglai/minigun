#ifndef SAMPLES_SAMPLES_IO_H_
#define SAMPLES_SAMPLES_IO_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>
#include "./samples_utils.h"
#include <iostream>

namespace utils {

// Csr graph that owns memory; easier for prototyping
struct SampleCsr {
  std::vector<int32_t> row_offsets;
  std::vector<int32_t> column_indices;
  int32_t num_rows;
  int32_t num_cols;
};

}  // namespace utils

namespace dmlc {
namespace serializer {

template <>
struct Handler<utils::SampleCsr> {
  inline static void Write(Stream* strm, const utils::SampleCsr& csr) {
    Handler<std::vector<int32_t>>::Write(strm, csr.row_offsets);
    Handler<std::vector<int32_t>>::Write(strm, csr.column_indices);
    Handler<int32_t>::Write(strm, csr.num_rows);
    Handler<int32_t>::Write(strm, csr.num_cols);
  }
  inline static bool Read(Stream* strm, utils::SampleCsr* out_csr) {
    if (!Handler<std::vector<int32_t>>::Read(strm, &(out_csr->row_offsets))) return false;
    if (!Handler<std::vector<int32_t>>::Read(strm, &(out_csr->column_indices))) return false;
    if (!Handler<int32_t>::Read(strm, &(out_csr->num_rows))) return false;
    if (!Handler<int32_t>::Read(strm, &(out_csr->num_cols))) return false;
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
  coo.row.length = n_edge;
  coo.column.length = n_edge;
  coo.num_rows = sample_csr.num_rows;
  coo.num_cols = sample_csr.num_cols;
  if (device == kDLCPU) {
    coo.row.data = new int32_t[n_edge];
    coo.column.data = new int32_t[n_edge];
    for (size_t row_id = 0; row_id < sample_csr.row_offsets.size() - 1; ++row_id) {
      for (size_t edge_id = sample_csr.row_offsets[row_id]; edge_id < sample_csr.row_offsets[row_id + 1]; ++edge_id) {
        coo.row.data[edge_id] = row_id;
        coo.column.data[edge_id] = sample_csr.column_indices[edge_id];
      }
    }
#ifdef __CUDACC__
  } else if (device == kDLGPU) {
    int32_t* row = new int32_t[n_edge];
    for (size_t row_id = 0; row_id < sample_csr.row_offsets.size() - 1; ++row_id) {
      for (size_t edge_id = sample_csr.row_offsets[row_id]; edge_id < sample_csr.row_offsets[row_id + 1]; ++edge_id) {
        row[edge_id] = row_id;
      }
    }
    CUDA_CALL(cudaMalloc(&coo.row.data, n_edge* sizeof(int32_t)));
    CUDA_CALL(cudaMemcpy(coo.row.data, row,
          sizeof(int32_t) * n_edge, cudaMemcpyHostToDevice));
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
  csr.row_offsets.length = rsize;
  csr.column_indices.length = csize;
  csr.num_rows = sample_csr.num_rows;
  csr.num_cols = sample_csr.num_cols;
  if (device == kDLCPU) {
    csr.row_offsets.data = new int32_t[rsize];
    std::copy(sample_csr.row_offsets.begin(), sample_csr.row_offsets.end(),
        csr.row_offsets.data);
    csr.column_indices.data = new int32_t[csize];
    std::copy(sample_csr.column_indices.begin(), sample_csr.column_indices.end(),
        csr.column_indices.data);
#ifdef __CUDACC__
  } else if (device == kDLGPU) {
    CUDA_CALL(cudaMalloc(&csr.row_offsets.data, rsize * sizeof(int32_t)));
    CUDA_CALL(cudaMemcpy(csr.row_offsets.data, &sample_csr.row_offsets[0],
          sizeof(int32_t) * rsize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&csr.column_indices.data, csize * sizeof(int32_t)));
    CUDA_CALL(cudaMemcpy(csr.column_indices.data, &sample_csr.column_indices[0],
          sizeof(int32_t) * csize, cudaMemcpyHostToDevice));
#endif  // __CUDACC__
  } else {
    LOG(INFO) << "Unsupported device: " << device;
  }
  return csr;
}

std::vector<int32_t*> transpose(const SampleCsr& sample_csr, int32_t* old_mapping_data) {
  // old_mapping_data is stored on CPU.
  const size_t n_v = sample_csr.row_offsets.size() - 1;
  const size_t n_e = sample_csr.column_indices.size();
  int32_t* row = new int32_t[n_v + 1];
  int32_t* col = new int32_t[n_e];
  int32_t* new_mapping_data = new int32_t[n_e];
  memset(row, 0, (n_v + 1) * sizeof(int32_t));
  memset(col, 0, (n_e) * sizeof(int32_t));
  memset(new_mapping_data, 0, (n_e) * sizeof(int32_t));
  // transpose
  // edge count
  for (size_t i = 0; i < n_e; ++i) {
    row[sample_csr.column_indices[i] + 1]++;
  }
  // cumsum on row pointer
  for (size_t i = 0; i < n_v; ++i) {
    row[i+1] += row[i];
  }
  // fill in edges
  for (size_t u = 0; u < n_v; ++u) {
    for (int32_t eid = sample_csr.row_offsets[u]; eid < sample_csr.row_offsets[u + 1]; ++eid) {
      int32_t v = sample_csr.column_indices[eid];
      new_mapping_data[row[v]] = old_mapping_data[eid];
      col[row[v]++] = u;
    }
  }
  // reset row pointer
  for (size_t u = n_v - 1; u > 0; --u) {
    row[u] = row[u - 1];
  }
  row[0] = 0;
  return {row, col, new_mapping_data};
}

std::pair<minigun::IntCsr, minigun::IntArray> ToMinigunReverseCsr(const SampleCsr& sample_csr, const minigun::IntArray& old_mapping, DLDeviceType device) {
  minigun::IntCsr csr;
  minigun::IntArray new_mapping;
  const size_t n_v = sample_csr.row_offsets.size() - 1;
  const size_t n_e = sample_csr.column_indices.size();
  int32_t* old_mapping_cpu = nullptr;
  if (old_mapping.length != n_e) {
    LOG(INFO) << "The length of mapping: does not equal number of edges in csr matrix";
  }
  if (device == kDLCPU) {
    old_mapping_cpu = old_mapping.data;
#ifdef __CUDACC__
  } else if (device == kDLGPU) {
    old_mapping_cpu = new int32_t[n_e];
    CUDA_CALL(cudaMemcpy(old_mapping_cpu, &old_mapping.data[0],
        sizeof(int32_t) * n_e, cudaMemcpyDeviceToHost));
#endif  // __CUDACC__
  } else {
    LOG(INFO) << "Unsupported device: " << device;
  }
  auto csr_t = transpose(sample_csr, old_mapping_cpu);
  int32_t* row = csr_t[0];
  int32_t* col = csr_t[1];
  int32_t* new_mapping_cpu = csr_t[2];
  csr.row_offsets.length = n_v + 1;
  csr.column_indices.length = n_e;
  new_mapping.length = n_e;
  csr.num_rows = sample_csr.num_cols;
  csr.num_cols = sample_csr.num_rows;
  if (device == kDLCPU) {
    csr.row_offsets.data = row;
    csr.column_indices.data = col;
    new_mapping.data = new_mapping_cpu;
#ifdef __CUDACC__
  } else if (device == kDLGPU) {
    CUDA_CALL(cudaMalloc(&csr.row_offsets.data, (n_v + 1) * sizeof(int32_t)));
    CUDA_CALL(cudaMemcpy(csr.row_offsets.data, &row[0],
          sizeof(int32_t) * (n_v + 1), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&csr.column_indices.data, n_e* sizeof(int32_t)));
    CUDA_CALL(cudaMemcpy(csr.column_indices.data, &col[0],
          sizeof(int32_t) * n_e, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&new_mapping.data, n_e * sizeof(int32_t)));
    CUDA_CALL(cudaMemcpy(new_mapping.data, &new_mapping_cpu[0],
          sizeof(int32_t) * n_e, cudaMemcpyHostToDevice));
#endif  // __CUDACC__
  } else {
    LOG(INFO) << "Unsupported device: " << device;
  }
  return {csr, new_mapping};
}

// create a sample csr that COPIES the memory of the minigun csr
__inline__ SampleCsr ToSampleCsr(const minigun::IntCsr& mg_csr, DLDeviceType device) {
  SampleCsr csr;
  csr.num_rows = mg_csr.num_rows;
  csr.num_cols = mg_csr.num_cols;
  csr.row_offsets.resize(mg_csr.row_offsets.length);
  csr.column_indices.resize(mg_csr.column_indices.length);
  if (device == kDLCPU) {
    std::copy(mg_csr.row_offsets.data, mg_csr.row_offsets.data + mg_csr.row_offsets.length,
              csr.row_offsets.begin());
    std::copy(mg_csr.column_indices.data, mg_csr.column_indices.data + mg_csr.column_indices.length,
              csr.column_indices.begin());
#ifdef __CUDACC__
  } else if (device == kDLGPU) {
    const int32_t n_v = mg_csr.row_offsets.length - 1;
    const int32_t n_e = mg_csr.column_indices.length;
    int32_t* row_offsets = new int32_t[n_v + 1];
    int32_t* column_indices = new int32_t[n_e];
    CUDA_CALL(cudaMemcpy(row_offsets, &mg_csr.row_offsets.data[0],
        (n_v + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(column_indices, &mg_csr.column_indices.data[0],
        n_e * sizeof(int32_t), cudaMemcpyDeviceToHost));
    std::copy(row_offsets, row_offsets + n_v + 1, csr.row_offsets.begin());
    std::copy(column_indices, column_indices + n_e, csr.column_indices.begin());
#endif  // __CUDACC__
  } else {
    LOG(INFO) << "Unsupported device: " << device;
  }
  return csr;
}

std::pair<minigun::IntCsr, minigun::IntArray> ToReverseCsr(
    const minigun::IntCsr& mg_csr,
    const minigun::IntArray old_mapping,
    DLDeviceType device) {
  SampleCsr scsr = ToSampleCsr(mg_csr, device);
  return ToMinigunReverseCsr(scsr, old_mapping, device);
}

minigun::IntCoo ToCoo(const minigun::IntCsr& mg_csr, DLDeviceType device) {
  SampleCsr scsr = ToSampleCsr(mg_csr, device);
  return ToMinigunCoo(scsr, device);
}

minigun::IntArray arange(int32_t low, int32_t high, DLDeviceType device) {
  if (low >= high) {
    LOG(INFO) << "low should not be greater than or equal to high";
  }
  minigun::IntArray rst;
  int32_t length = high - low;
  int32_t *rst_data = new int32_t[length];
  for (int32_t i = 0; i < length; ++i) {
    rst_data[i] = low + i;
  }
  rst.length = length;
  if (device == kDLCPU) {
    rst.data = rst_data;
#ifdef __CUDACC__
  } else if (device == kDLGPU) {
    CUDA_CALL(cudaMalloc(&rst.data, length * sizeof(int32_t)));
    CUDA_CALL(cudaMemcpy(rst.data, &rst_data[0],
        length * sizeof(int32_t), cudaMemcpyHostToDevice));
#endif  // __CUDACC__
  } else {
    LOG(INFO) << "Unsupported device: " << device;
  }
  return rst;
}

}  // namespace utils

#endif  // SAMPLES_SAMPLES_IO_H_
