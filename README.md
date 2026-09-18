# umap

`umap` is an allocation-conscious Go implementation of Uniform
Manifold Approximation and Projection (UMAP).

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

The proposed v1 Go API, pinned Python compatibility target, and numerical
guarantees are specified in [the v1 API and compatibility contract](docs/v1-contract.md).
Scalable search controls, memory accounting, and benchmark methodology are in
[the Milestone 2 neighbor-search guide](docs/m2-neighbor-search.md).
