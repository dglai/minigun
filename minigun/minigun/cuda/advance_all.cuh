#ifndef MINIGUN_CUDA_ADVANCE_ALL_CUH_
#define MINIGUN_CUDA_ADVANCE_ALL_CUH_

namespace minigun {
namespace advance {

// executor for advance all
template <typename GData,
          typename Functor>
struct AdvanceExecutor<StaticConfig<true>, GData, Functor> {
  static void Run(csr, gdata, input_frontier, output_frontier) {
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_ALL_CUH_
