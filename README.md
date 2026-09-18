# umap

`umap` is an allocation-conscious Go implementation of Uniform
Manifold Approximation and Projection (UMAP).

The deterministic single-threaded core currently includes:

- validated zero-copy row-major dense and canonical CSR matrices;
- dense and sparse distance evaluation, exact neighbors, smooth-kNN distances,
  and compact fuzzy graph construction;
- random and spectral initialization plus reference layout optimization; and
- deterministic seed derivation and sampling independent of architecture.
- blocked exact and deterministic NN-descent neighbor search with cancellation,
  progress reporting, and an explicit graph memory model;
- allocation-free CSR distance kernels that keep sparse inputs sparse through
  graph construction.

The proposed v1 Go API, pinned Python compatibility target, and numerical
guarantees are specified in [the v1 API and compatibility contract](docs/v1-contract.md).
Scalable search controls, memory accounting, and benchmark methodology are in
[the Milestone 2 neighbor-search guide](docs/m2-neighbor-search.md).
