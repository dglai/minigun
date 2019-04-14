#ifndef MINIGUN_CSR_H_
#define MINIGUN_CSR_H_

#include <dlpack/dlpack.h>
#include "./base.h"

namespace minigun {

// NOTE: minigun does not own any memory

struct IntArray1D {
  mg_int* data = nullptr;
  mg_int length = 0;
};

// edges of node i are stored in column_indices[row_offsets[i]:row_offsets[i+1]]
struct Csr {

  IntArray1D row_offsets;  // len == num_nodes + 1
  IntArray1D column_indices;  // len == num_edges

};

}  // namespace minigun

#endif  // MINIGUN_CSR_H_
