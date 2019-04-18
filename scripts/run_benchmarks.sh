#!/bin/bash

# The script should be called in the root directory of the repo.
# Assume the binary has been built in `./build`

if [ ! -d "bench-graphs" ]; then
  echo "Please generate benchmark graphs first by scripts/gen_benchmark_graphs.sh"
  exit 1
fi

if [ ! -d "build" ]; then
  echo "Please build the samples first to 'build' folder"
  exit 1
fi

# masked mm test
echo "===========Masked MM test==========="
for grh in bench-graphs/*;do
  for D in 8 32 64 128; do
    for H in 8 16 32; do
      ./build/samples/benchmark/bench_masked_mm ${grh} $D $H || exit 1
    done
  done
done

# edge softmax test
echo "===========Edge softmax test==========="
for grh in bench-graphs/*;do
  for H in 8 16 32; do
    ./build/samples/benchmark/bench_edge_softmax ${grh} $H || exit 1
  done
done

# spmm test
echo "===========SPMM test==========="
for grh in bench-graphs/*;do
  for D in 8 32 64 128; do
    for H in 8 16 32; do
      ./build/samples/benchmark/bench_spmm ${grh} $D $H || exit 1
    done
  done
done

# backward edge softmax test
echo "===========Backward edge softmax test==========="
for grh in bench-graphs/*;do
  for H in 8 16 32; do
    ./build/samples/benchmark/bench_backward_edge_softmax ${grh} $H || exit 1
  done
done
