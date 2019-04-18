import networkx as nx
import argparse

import graph_io

def gen_er(args):
    g = nx.fast_gnp_random_graph(args.er_n, args.er_p)
    csr = nx.to_scipy_sparse_matrix(g, weight=None, format='csr')
    graph_io.save_graph(args.out, csr)

def gen_ba(args):
    g = nx.barabasi_albert_graph(args.ba_n, args.ba_m)
    csr = nx.to_scipy_sparse_matrix(g, weight=None, format='csr')
    graph_io.save_graph(args.out, csr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('generator', type=str, help='er|ba')
    parser.add_argument('-o', '--out', type=str, help='Output file name')
    parser.add_argument('-n', '--nodes', type=str, help='Number of nodes')
    parser.add_argument('--er-n', type=int, default=10000, help='n in E-R graph')
    parser.add_argument('--er-p', type=float, default=0.001, help='p in E-R graph')
    parser.add_argument('--ba-n', type=int, default=10000, help='n in B-A graph')
    parser.add_argument('--ba-m', type=int, default=10, help='m in B-A graph')
    args = parser.parse_args()
    print(args)
    if args.generator == 'er':
        gen_er(args)
    elif args.generator == 'ba':
        gen_ba(args)
    else:
        print('Unknown generator:', args.generator)
