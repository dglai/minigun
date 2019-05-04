# MiniGun: Light-weight GPU kernel interface for graph operations

The project is heavily influenced by the [Gunrock project](https://github.com/gunrock/gunrock).
The goal is to define a general kernel library that can cover a wide range of graph operations used in [DGL](https://github.com/dmlc/dgl).

Current roadmap:
- [x] Port basic advance operator (all edge policy).
- [x] Port advance operator with dynamic frontiers.
- [ ] Port more advance kernel policies
  - [ ] TWC
  - [ ] Inbound edge partition
  - [x] outbound edge partition
- [x] CPU implementation for advance
- [x] Integration with DGL
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

-------------
*Disclaimer:* Minigun project is not related to the ["mini-gunrock" project](https://github.com/gunrock/mini).
