import networkx as nx
import argparse
import dgl
from dgl.data import register_data_args, load_data
from segtree import build_segtree

import graph_io

def main(args):
    if args.dataset == 'segtree':
        g = build_segtree(batch_size=32, seq_len=512)
        print('#Nodes: %d #Edges: %d' % (g.number_of_nodes(), g.number_of_edges()))
        csr = g.adjacency_matrix_scipy(fmt='csr')        
    else:
        data = load_data(args)
        g = data.graph
        csr = nx.to_scipy_sparse_matrix(g, weight=None, format='csr')

    graph_io.save_graph(args.out, csr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', type=str, help='Output file name')
    register_data_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
