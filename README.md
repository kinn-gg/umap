# umap

`umap` is a production-oriented, allocation-conscious Go implementation of
Uniform Manifold Approximation and Projection (UMAP) for Go 1.24 and newer.

The deterministic single-threaded implementation includes:

- validated zero-copy row-major dense and canonical CSR matrices;
- dense and sparse distance evaluation, exact neighbors, smooth-kNN distances,
  and compact fuzzy graph construction;
- random and spectral initialization plus reference layout optimization; and
- deterministic seed derivation and sampling independent of architecture.
- blocked exact and deterministic NN-descent neighbor search with cancellation,
  progress reporting, and an explicit graph memory model;
- allocation-free CSR distance kernels that keep sparse inputs sparse through
  graph construction.
- reusable fit/transform models for dense and sparse observations;
- categorical, semi-supervised, and continuous target-informed fitting; and
- checksummed, versioned model serialization with bounded decoding.
- densMAP density preservation and optional original/embedding density radii.

```go
cfg := umap.DefaultConfig()
seed := uint64(7)
cfg.Seed, cfg.Deterministic = &seed, true
reducer, err := umap.New(cfg)
if err != nil { log.Fatal(err) }

x, err := umap.NewDense(samples, rows, columns)
if err != nil { log.Fatal(err) }
model, embedding, err := reducer.FitTransform(context.Background(), x, umap.NoTarget())
if err != nil { log.Fatal(err) }

query, err := umap.NewDense(newSamples, newRows, columns)
if err != nil { log.Fatal(err) }
projected, err := model.Transform(context.Background(), query)
if err != nil { log.Fatal(err) }
fmt.Println(embedding.At(0, 0), projected.At(0, 0))
```

Models implement `encoding.BinaryMarshaler`; restore them with
`umap.UnmarshalModel`. The format includes learned graph, embedding, owned
training/search data, parameters, a version marker, and a SHA-256 checksum.

Enable densMAP with `cfg.Density.Enabled = true`. Set
`cfg.Density.Output = true` to retrieve immutable log-radius vectors through
`Model.OriginalRadii` and `Model.EmbeddingRadii`; output-only mode does not
change the embedding.

## Documentation

- [Getting started, tuning, migration, and serialization](docs/guide.md)
- [Supported-feature compatibility matrix and known differences](docs/compatibility.md)
- [API contract and numerical tolerances](docs/v1-contract.md)
- [Reproducible benchmark results and regression policy](docs/m4-performance.md)
- [Scalable search controls and memory accounting](docs/m2-neighbor-search.md)
- [Release, semantic-versioning, and attribution policy](docs/releasing.md)

The implementation follows the UMAP and densMAP papers and pins compatibility
claims to `umap-learn` 0.5.12. Coordinates are not expected to be byte-identical
across implementations; the contract defines artifact and quality tolerances.

## License

BSD 3-Clause. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
