import numpy as np
import dgl
from scipy.sparse import coo_matrix

def overlap(i, j, left=True):
    if j == 0:
        return False
    if left:
        return (i >> (j - 1)) & 1 == 0
    else:
        return (i >> (j - 1)) & 1 == 1

class SegmentTree:
    def __init__(self, length):
        self.n_nodes = 0
        self.n_edges = 0
        self.length = length
        self.n_lvl = 1
        self.shift = []
        self.edges = [[] for _ in range(2)]
        self.n_nodes_arr = []
        self.build_graph()

    def build_graph(self):
        i = self.length
        self.shift.append(0)
        while i >= 2:
            self.n_nodes_arr.append(i)
            self.n_nodes += i
            self.shift.append(self.n_nodes)
            self.n_lvl += 1
            i = (i + 1) >> 1
        
        # root
        self.n_nodes_arr.append(1)
        self.n_nodes += 1
        
        # add edges
        for i in range(self.length):
            v, shift = i, 0
            for j in range(self.n_lvl):
                # Self-loop
                if (i == (v << j)):
                    self.edges[0].append(shift + v)
                    self.edges[1].append(shift + v)

                if i + (1 << j) < self.length:
                    if overlap(i, j, left=False):
                        self.edges[0].append(self.shift[j - 1] + (((v + 1) << 1) + 1))
                    else:
                        self.edges[0].append(self.shift[j] + (v + 1))
                    self.edges[1].append(i)
                    self.n_edges += 1
                
                if v >= 1:
                    if overlap(i, j, left=True):
                        self.edges[0].append(self.shift[j - 1] + ((v - 1) << 1))
                    else:
                        self.edges[0].append(self.shift[j] + (v - 1))
                    self.edges[1].append(i)
                    self.n_edges += 1

                if j > 0:
                    self.edges[0].append(i)
                    self.edges[1].append(shift + v)
                    self.n_edges += 1

                shift += self.n_nodes_arr[j]
                v >>= 1

    def get_edges(self, v_shift=0):
        return np.array(self.edges[0]), np.array(self.edges[1])

def build_segtree(batch_size=1, seq_len=70):
    t = SegmentTree(seq_len)
    v_shift = 0
    row, col = [], []
    for _ in range(batch_size):
        edges = t.get_edges(v_shift=v_shift)
        row.append(edges[0])
        col.append(edges[1])
        v_shift += t.n_nodes
        
    row, col = map(np.concatenate, (row, col))
    coo = coo_matrix((np.zeros_like(row), (row, col)), shape=(v_shift, v_shift))
    g = dgl.DGLGraph(coo, readonly=True)
    return g

