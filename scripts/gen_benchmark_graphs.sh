#!/bin/bash

# The script should be called in the root directory of the repo.

mkdir -p bench-graphs

# generate graphs
echo "Citation graphs..."
python3 scripts/gen_dgl_graph.py -o bench-graphs/x_cora.grh --dataset=cora
python3 scripts/gen_dgl_graph.py -o bench-graphs/y_citeseer.grh --dataset=citeseer
python3 scripts/gen_dgl_graph.py -o bench-graphs/z_pubmed.grh --dataset=pubmed
python3 scripts/gen_dgl_graph.py -o bench-graphs/reddit.grh --dataset=reddit-self-loop

# generate E-R graphs
ER_N=( 10000 10000 100000 100000 )
ER_P=( 0.001 0.01 0.00001 0.0001 )
echo "E-R graphs..."
for ((i=0;i<${#ER_N[@]};++i)); do
    n=${ER_N[i]}
    p=${ER_P[i]}
    echo "N=$n P=$p"
    python3 scripts/gen_rand_graph.py er -o bench-graphs/er_${i}.grh --er-n=$n --er-p=$p
done

# generate B-A graphs
BA_N=( 10000 10000 100000 100000 )
BA_M=( 5 50 1 5 )
echo "B-A graphs..."
count=0
for ((i=0;i<${#BA_N[@]};++i)); do
    n=${BA_N[i]}
    m=${BA_M[i]}
    echo "N=$n M=$m"
    python3 scripts/gen_rand_graph.py ba -o bench-graphs/ba_${i}.grh --ba-n=$n --ba-m=$m
done
