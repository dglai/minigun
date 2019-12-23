import struct
import numpy as np
import scipy.sparse as sp

# return a scipy.sparse.csr_matrix
def load_graph(filename):
    with open(filename, 'rb') as f:
        l = struct.unpack('q', f.read(8))[0]
        indptr = np.frombuffer(f.read(l * 4), dtype=np.int32)
        l = struct.unpack('q', f.read(8))[0]
        indices = np.frombuffer(f.read(l * 4), dtype=np.int32)
        n_row = struct.unpack('i', f.read(4))[0]
        n_col = struct.unpack('i', f.read(4))[0]
        N = indptr.shape[0] - 1
        assert n_row == N, "Size does not match."
        M = indices.shape[0]
        data = np.ones((M,), dtype=np.float32)
        return sp.csr_matrix((data, indices, indptr), shape=(n_row, n_col), dtype=np.float32)

# assume mg_int is int32_t
# g is in scipy.sparse.csr_matrix
def save_graph(filename, g, n, m):
    with open(filename, 'wb') as f:
        f.write(struct.pack('q', g.indptr.shape[0]))
        f.write(g.indptr.astype(np.int32).tobytes())
        f.write(struct.pack('q', g.indices.shape[0]))
        f.write(g.indices.astype(np.int32).tobytes())
        f.write(struct.pack('i', int(n)))
        f.write(struct.pack('i', int(m)))

