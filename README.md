# umap

`umap` is an allocation-conscious Go implementation of Uniform
Manifold Approximation and Projection (UMAP).

The deterministic single-threaded core currently includes:

- validated zero-copy row-major dense and canonical CSR matrices;
- dense and sparse distance evaluation, exact neighbors, smooth-kNN distances,
  and compact fuzzy graph construction;
- random and spectral initialization plus reference layout optimization; and
- deterministic seed derivation and sampling independent of architecture.

The proposed v1 Go API, pinned Python compatibility target, and numerical
guarantees are specified in [the v1 API and compatibility contract](docs/v1-contract.md).
