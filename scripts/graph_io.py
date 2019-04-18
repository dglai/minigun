import struct
import numpy as np
import scipy.sparse as sp

# assume mg_int is int64_t
# return a scipy.sparse.csr_matrix
# NOTE: scipy currently does not support int64 indices, all int64 will be casted to int32
def load_graph(filename):
    with open(filename, 'rb') as f:
        l = struct.unpack('q', f.read(8))[0]
        indptr = np.frombuffer(f.read(l * 8), dtype=np.int64)
        l = struct.unpack('q', f.read(8))[0]
        indices = np.frombuffer(f.read(l * 8), dtype=np.int64)
        N = indptr.shape[0] - 1
        M = indices.shape[0]
        data = np.ones((M,), dtype=np.float32)
        return sp.csr_matrix((data, indices, indptr), shape=(N, N), dtype=np.float32)

# assume mg_int is int64_t
# g is in scipy.sparse.csr_matrix
# NOTE: scipy currently does not support int64 indices, all int64 will be casted to int32
def save_graph(filename, g):
    with open(filename, 'wb') as f:
        f.write(struct.pack('q', g.indptr.shape[0]))
        f.write(g.indptr.astype(np.int64).tobytes())
        f.write(struct.pack('q', g.indices.shape[0]))
        f.write(g.indices.astype(np.int64).tobytes())
