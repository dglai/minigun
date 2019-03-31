# MiniGun: Light-weight GPU kernel interface for graph operations

The project is heavily influenced by the Gunrock project. The goal is to define a general kernel library
that can cover a wide range of graph operations.

Roadmap:
- [x] Port basic advance operator (all edge policy).
- [ ] Port advance operator with dynamic frontiers.
- [ ] Port more advance kernel policies
  - [ ] TWC
  - [ ] Inbound edge partition
  - [ ] outbound edge partition
- [ ] CPU implementation for advance
- [ ] Integration with DGL
- [ ] Filter

## Build
Require CUDA.
```bash
mkdir build
cmake ..
make -j4
```

## Play
See codes in samples. Currently, there are four:

* spmv
* spmm
* masked_mm
* edge_softmax

You could run sample by (in the `build` folder):
```bash
./samples/spmv/spmv
```
