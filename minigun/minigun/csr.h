#ifndef MINIGUN_CSR_H_
#define MINIGUN_CSR_H_

#include <dlpack/dlpack.h>
#include "./base.h"

namespace minigun {

// NOTE: minigun does not own any memory

template <typename Idx>
struct IntArray1D {
  Idx* data = nullptr;
  int64_t length = 0;
};

using IntArray = IntArray1D<int32_t>;
using LongArray = IntArray1D<int64_t>;

// edges of node i are stored in column_indices[row_offsets[i]:row_offsets[i+1]]
template <typename Idx>
struct Csr {
  IntArray1D<Idx> row_offsets;  // len == num_nodes + 1
  IntArray1D<Idx> column_indices;  // len == num_edges
};

using IntCsr = Csr<int32_t>;
using LongCsr = Csr<int64_t>;

}  // namespace minigun

#endif  // MINIGUN_CSR_H_
