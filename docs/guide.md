# Guide: fit, transform, serialize, and tune

## Fit and transform

Create a row-major `[]float32`, validate it with `NewDense`, start from
`DefaultConfig`, then call `FitTransform`. Pass `NoTarget()` for unsupervised
learning. A fitted `Model` owns its training data, so the caller may reuse the
input after fitting.

```go
seed := uint64(7)
cfg := umap.DefaultConfig()
cfg.Seed, cfg.Deterministic = &seed, true
reducer, err := umap.New(cfg)
if err != nil { return err }
x, err := umap.NewDense(training, rows, columns)
if err != nil { return err }
model, embedding, err := reducer.FitTransform(ctx, x, umap.NoTarget())
if err != nil { return err }

query, err := umap.NewDense(unseen, queryRows, columns)
if err != nil { return err }
projected, err := model.Transform(ctx, query)
```

CSR inputs use `NewCSR(values, columns, offsets, rows, featureCount)` and stay
sparse. Supervised fitting uses `NewCategories` or `NewContinuous`.

## Save and restore

```go
encoded, err := model.MarshalBinary()
if err != nil { return err }
if err := os.WriteFile("model.umap", encoded, 0o600); err != nil { return err }

encoded, err = os.ReadFile("model.umap")
if err != nil { return err }
model, err = umap.UnmarshalModel(encoded)
```

The format is checksummed and bounded but is not Python pickle-compatible.
Treat models as application data, not as a substitute for versioned source
fixtures. Decoders accept the previous format version.

## Tuning

Start with defaults. Increase `Neighbors` (30–100) to emphasize broad manifold
structure, or reduce it (5–15) for local structure. Lower `MinDist` produces
tighter clusters; higher values spread points more evenly. Use approximate
neighbor search for larger data and set `MemoryBudget` where peak memory must be
bounded. A fixed `Seed`, `Deterministic=true`, and one effective worker provide
the strongest reproducibility guarantee.

For density preservation, set `Density.Enabled=true`. `Density.Lambda` controls
the density objective, `Fraction` selects the final fraction of epochs where it
runs, and `VarianceShift` stabilizes low-variance data. `Density.Output=true`
records log radii even without changing layout. densMAP costs additional runtime
and `O(rows)` scratch only when density functionality is requested.

## Python migration

`UMAP.fit_transform(X)` maps to `FitTransform(ctx, x, NoTarget())` and returns a
model plus its embedding. `fit(X)` maps to `Fit`; `transform(X)` maps to
`Model.Transform`. Go configuration fields replace Python keyword arguments,
and inputs are explicit `float32` dense or CSR values. See the
[compatibility matrix](v1-contract.md#defaults-and-python-parameter-mapping) for
each parameter and the documented numerical differences.
