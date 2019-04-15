#include <minigun/minigun.h>
#include "../samples_io.h"
#include "../samples_utils.h"

int main(int argc, char** argv) {
  srand(42);
  utils::SampleCsr csr, new_csr;
  utils::CreateNPGraph(10000, 0.01, csr.row_offsets, csr.column_indices);
  utils::SaveGraphToFile("test.grh", csr);
  assert(utils::LoadGraphFromFile("test1.grh", &new_csr));
  LOG(INFO) << "#Nodes: " << new_csr.row_offsets.size() - 1 << " #Edges: " << new_csr.column_indices.size();
  assert(utils::VecEqual(csr.row_offsets, new_csr.row_offsets));
  assert(utils::VecEqual(csr.column_indices, new_csr.column_indices));
}
