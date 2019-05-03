#!/bin/bash

# The script should be called in the root directory of the repo.

ER_N=( 1000 5000 10000 )
ER_P=( 0.01 0.005 0.001 )
BA_N=( 1000 5000 10000 )
BA_M=( 5 10 20 )

mkdir -p bench-graphs

# generate graphs
echo "Citation graphs..."
python3 scripts/gen_dgl_graph.py -o bench-graphs/cora.grh --dataset=cora
python3 scripts/gen_dgl_graph.py -o bench-graphs/citeseer.grh --dataset=citeseer
python3 scripts/gen_dgl_graph.py -o bench-graphs/pubmed.grh --dataset=pubmed
python3 scripts/gen_dgl_graph.py -o bench-graphs/segtree.grh --dataset=segtree

# generate E-R graphs
echo "E-R graphs..."
count=0
for n in "${ER_N[@]}"
do
  for p in "${ER_P[@]}"
  do
    echo "N=$n P=$p"
    python3 scripts/gen_rand_graph.py er -o bench-graphs/er_${count}.grh --er-n=$n --er-p=$p
    count=$(( $count + 1 ))
  done
done

# generate B-A graphs
echo "B-A graphs..."
count=0
for n in "${BA_N[@]}"
do
  for m in "${BA_M[@]}"
  do
    echo "N=$n M=$m"
    python3 scripts/gen_rand_graph.py ba -o bench-graphs/ba_${count}.grh --ba-n=$n --ba-m=$m
    count=$(( $count + 1 ))
  done
done
