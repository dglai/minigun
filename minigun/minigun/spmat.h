#ifndef MINIGUN_CSR_H_
#define MINIGUN_CSR_H_

#include <dlpack/dlpack.h>
#include "./base.h"

namespace minigun {

// NOTE: minigun does not own any memory

template <typename Idx>
struct IntArray1D {
  Idx* data = nullptr;
  Idx length = 0;
};

using IntArray = IntArray1D<int32_t>;
using LongArray = IntArray1D<int64_t>;

// edges of node i are stored in column_indices[row_offsets[i]:row_offsets[i+1]]
template <typename Idx>
struct Csr {
  IntArray1D<Idx> row_offsets;  // len == num_nodes + 1
  IntArray1D<Idx> column_indices;  // len == num_edges
  Idx num_rows;
  Idx num_cols;
};

using IntCsr = Csr<int32_t>;
using LongCsr = Csr<int64_t>;

template <typename Idx>
struct Coo {
  IntArray1D<Idx> row;
  IntArray1D<Idx> column;
  Idx num_rows;
  Idx num_cols;
};
using IntCoo = Coo<int32_t>;
using LongCoo = Coo<int64_t>;

template <typename Idx>
struct SpMat {
  Csr<Idx>* csr = nullptr;
  Csr<Idx>* csr_t = nullptr;
  Coo<Idx>* coo = nullptr;
  SpMat(Csr<Idx>* csr, Csr<Idx>* csr_t, Coo<Idx>* coo):
    csr(csr), csr_t(csr_t), coo(coo) {}
};
using IntSpMat = SpMat<int32_t>;
using LongSpMat = SpMat<int64_t>;

}  // namespace minigun

#endif  // MINIGUN_CSR_H_
