/* Sample code for Dense-Dense => Sparse Matrix multiplication.*/
#include <iostream>
#include <cstdlib>
#include <time.h>

#include <minigun/minigun.h>
#include "../benchmark_utils.h"

int main(int argc, char** argv) {
  srand(42);

  std::vector<mg_int> row_offsets;
  std::vector<mg_int> column_indices;
  
  utils::CreateFullBatch0(5, row_offsets, column_indices);
  utils::VecPrint(row_offsets);
  utils::VecPrint(column_indices);

  row_offsets.clear();
  column_indices.clear();
  utils::CreateSparseBatch0(5, row_offsets, column_indices);
  utils::VecPrint(row_offsets);
  utils::VecPrint(column_indices);
  return 0;
}
