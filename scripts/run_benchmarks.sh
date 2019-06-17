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

# spmm test
echo "===========SPMM test==========="
for grh in bench-graphs/*;do
  for D in 8 16 32 64 128 256 512; do
    ./build/samples/benchmark/bench_spmm ${grh} $D || exit 1
  done
done
