# Compatibility and known differences

The compatibility target is `umap-learn` 0.5.12. The differential harness pins
its Python, NumPy, Numba, and PyNNDescent environment in `parity/uv.lock`.

| Capability | v1 support | Notes |
|---|---|---|
| Dense input | Yes | Row-major `float32`, optional stride |
| Sparse input | Yes | Canonical CSR; never densified |
| Metrics | Yes | Euclidean, squared Euclidean, Manhattan, Chebyshev, Minkowski, cosine, correlation, Canberra, Bray-Curtis, Hamming, Jaccard, Dice |
| Exact/approximate neighbors | Yes | Deterministic tie ordering and bounded memory |
| Random/spectral initialization | Yes | PCA, tswspectral, and caller coordinates are deferred |
| Fit and transform | Yes | Graph transform mode is deferred |
| Categorical/continuous supervision | Yes | Unknown categorical labels support semi-supervision |
| densMAP/output density | Yes | Immutable original and embedding log radii |
| Model persistence | Yes | Versioned Go format; not Python pickle-compatible |
| Custom/precomputed metrics or kNN | Deferred | Requests are not silently approximated |

Exact neighbors, smooth-kNN values, and fuzzy graph weights use the tolerances in
the [v1 contract](v1-contract.md#differential-correctness-gates). Final
coordinates can rotate, reflect, or differ in low bits because eigensolvers,
floating-point evaluation, and RNG implementations differ. Quality is compared
through neighborhood preservation and density correlation rather than coordinate
identity. Deterministic mode promises repeatability only for the same package,
toolchain, architecture, and CPU features.
