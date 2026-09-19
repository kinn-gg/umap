# v1 API and `umap-learn` compatibility contract

Status: **v1 contract**

Last updated: 2026-09-18

This document defines the v1 surface and compatibility expectations.

## Reference and meaning of compatibility

The reference implementation is
[`lmcinnes/umap` release 0.5.12](https://github.com/lmcinnes/umap/releases/tag/release-0.5.12),
commit [`cdb0eeb236f5d6759e0c3dd1fe132aef4db9e7b1`](https://github.com/lmcinnes/umap/commit/cdb0eeb236f5d6759e0c3dd1fe132aef4db9e7b1).
Reference fixtures must record that commit, the Python/NumPy/Numba/
PyNNDescent versions, platform, input checksum, parameters, and seeds.

Compatibility means matching documented `UMAP` semantics, validated
intermediate artifacts, and invariant embedding quality. It does **not** mean
that coordinates must be element-for-element equal: eigensolvers, floating
point contraction, neighbor-search traversal, and parallel scheduling can
rotate, reflect, or slightly deform an otherwise equivalent embedding.

## Package surface

```go
package umap

type Config struct {
    Neighbors          int
    Components         int
    Metric             Metric
    OutputMetric       OutputMetric
    Epochs             int             // 0 selects the reference heuristic
    LearningRate       float32
    Init                Init
    MinDist             float32
    Spread              float32
    SetOpMixRatio       float32
    LocalConnectivity   float32
    RepulsionStrength   float32
    NegativeSampleRate int
    A, B                float32         // both 0 means fit from MinDist/Spread
    Seed                *uint64         // nil requests a nondeterministic seed
    TransformSeed       uint64
    Workers             int             // 0 selects GOMAXPROCS
    Deterministic       bool            // requires Workers to be 0 or 1
    Search              SearchConfig
    Target              TargetConfig
    Density             DensityConfig
    DisconnectionDistance float32       // +Inf by default
}

func DefaultConfig() Config
func New(config Config) (*UMAP, error)

func (u *UMAP) Fit(ctx context.Context, x Matrix, y Target) (*Model, error)
func (u *UMAP) FitTransform(ctx context.Context, x Matrix, y Target) (*Model, *Embedding, error)

func (m *Model) Embedding() *Embedding
func (m *Model) Transform(ctx context.Context, x Matrix) (*Embedding, error)
func (m *Model) TransformInto(ctx context.Context, dst []float32, x Matrix) error

type Matrix interface {
    Shape() (rows, columns int)
    matrix() // sealed: implemented by Dense and CSR supplied by this package
}

func NewDense(data []float32, rows, columns int) (Dense, error)
func NewDenseStride(data []float32, rows, columns, stride int) (Dense, error)
func NewCSR(values []float32, columns []uint32, rowOffsets []uint64,
    rows, columnCount int) (CSR, error)

type Target interface { target() /* sealed: NoTarget, Categories, Continuous */ }
func NoTarget() Target
func NewCategories(values []int32, unknown int32) (Categories, error)
func NewContinuous(values []float32) (Continuous, error)

func (e *Embedding) Shape() (rows, components int)
func (e *Embedding) At(row, component int) float32
func (e *Embedding) CopyTo(dst []float32) error

func (m *Model) OriginalRadii() (*Vector, bool)
func (m *Model) EmbeddingRadii() (*Vector, bool)
func (v *Vector) Len() int
func (v *Vector) At(index int) float32
func (v *Vector) CopyTo(dst []float32) error
```

`Metric`, `OutputMetric`, `Init`, and search/target/density configurations are
closed, typed values for built-ins. A separate custom metric interface may be
added only if benchmarks show it can avoid allocation and interface dispatch in
the inner loop. `New` validates all cross-field constraints and retains a copy
of `Config`; changing the caller's value after `New` has no effect.

### Unsupervised dense example

```go
cfg := umap.DefaultConfig()
seed := uint64(7)
cfg.Seed = &seed
cfg.Deterministic = true

reducer, err := umap.New(cfg)
if err != nil { log.Fatal(err) }

x, err := umap.NewDense(samples, rows, columns) // samples is []float32
if err != nil { log.Fatal(err) }

model, embedding, err := reducer.FitTransform(context.Background(), x, umap.NoTarget())
if err != nil { log.Fatal(err) }

fmt.Println(embedding.At(0, 0), embedding.At(0, 1))
_ = model
```

### Sparse fit and transform into caller memory

```go
x, err := umap.NewCSR(values, columnIndices, rowOffsets, rows, columns)
if err != nil { log.Fatal(err) }

reducer, err := umap.New(umap.DefaultConfig())
if err != nil { log.Fatal(err) }
model, err := reducer.Fit(ctx, x, umap.NoTarget())
if err != nil { log.Fatal(err) }

query, err := umap.NewCSR(qValues, qColumns, qRows, queryRows, columns)
if err != nil { log.Fatal(err) }
out := make([]float32, queryRows*2)
if err := model.TransformInto(ctx, out, query); err != nil { log.Fatal(err) }
```

### Supervised fit

```go
labels, err := umap.NewCategories(classes, -1) // -1 is unlabeled
if err != nil { log.Fatal(err) }
model, embedding, err := reducer.FitTransform(ctx, x, labels)
```

## Data, precision, and ownership

- v1 accepts row-major `float32` dense data and canonical CSR data. CSR column
  indices in each row must be strictly increasing, in range, and duplicate-free;
  row offsets must be monotonic and end at `len(values)`. Dense strides may
  include padding but may not overlap rows. Empty feature dimensions are invalid.
- Constructors validate slice lengths and integer conversions without copying.
  They retain read-only views of caller buffers. The caller must not mutate or
  resize an input buffer until the operation using it returns. `Fit` may retain
  training data for later `Transform`; when it does, the `Model` owns an internal
  compact copy, so caller buffers may be reused after `Fit` returns.
- Sparse input is never densified. Stage-local memory is bounded in terms of
  nonzeros plus `O(rows * Neighbors)` unless a documented algorithm requires
  otherwise.
- Storage and public outputs are `float32`. Distance reductions, normalization,
  curve fitting, and other cancellation-sensitive accumulations use `float64`
  before a checked conversion to `float32`. v1 does not silently convert
  `float64` inputs; callers choose and perform that conversion explicitly.
- `Embedding` is immutable and may be shared safely. `FitTransform` returns the
  same read-only embedding held by `Model`, avoiding an output copy.
  `Transform` allocates a new immutable embedding. `TransformInto` performs no
  output allocation, requires exactly `queryRows * Components` elements, and
  leaves `dst` unspecified if it returns an error.
- Accessors for learned graph/search state return immutable views or copy into a
  caller buffer. No public method exposes mutable model storage.

## Lifecycle, cancellation, errors, and concurrency

- `UMAP` is immutable after `New`. Its `Fit` and `FitTransform` methods may run
  concurrently and use per-call scratch state.
- `Model` is immutable. `Embedding`, `Transform`, and `TransformInto` may run
  concurrently. Concurrent calls never share mutable RNG or scratch buffers.
  As with all Go slice APIs, the caller must not use the same `dst` slice in
  concurrent `TransformInto` calls.
- Every operation that can perform unbounded work accepts `context.Context`.
  It checks cancellation between bounded distance blocks, neighbor-descent
  iterations, graph stages, and optimization epochs. It returns an error for
  which `errors.Is(err, context.Canceled)` or
  `errors.Is(err, context.DeadlineExceeded)` succeeds. Cancellation never leaks
  workers or publishes a partial model.
- Invalid configuration, shape, numeric input, CSR structure, unfitted/model
  state, and unsupported combinations return exported sentinel-compatible typed
  errors (`ValidationError`, `ShapeError`, `NumericError`, `UnsupportedError`).
  User data must not cause a panic.
- NaN and infinity are rejected in v1. Integer products and index conversions
  are checked before allocation. Inputs too large for the selected backend fail
  with a typed error rather than wrapping or truncating.

## Defaults and Python parameter mapping

“v1” identifies behavior covered by the compatibility contract. “Deferred”
means it is not silently approximated; requesting the behavior returns
`UnsupportedError`.
Options omitted from the Go API because they are automatic or replaced by a Go
mechanism are still listed.

| Python 0.5.12 parameter (default) | Go mapping/default | Status and notes |
|---|---|---|
| `n_neighbors=15` | `Config.Neighbors=15` | v1 |
| `n_components=2` | `Config.Components=2` | v1 |
| `metric='euclidean'` | `Config.Metric=Euclidean` | v1 built-ins: Euclidean, squared Euclidean, Manhattan, Chebyshev, Minkowski, cosine, correlation, Canberra, Bray-Curtis, Hamming, Jaccard, Dice. Remaining Python metrics deferred. |
| `metric_kwds=None` | typed fields on `Metric` (for example Minkowski `P`) | v1 for supported metrics; no order-sensitive string map |
| `output_metric='euclidean'` | `Config.OutputMetric=OutputEuclidean` | v1 Euclidean; other output metrics and callable gradients deferred |
| `output_metric_kwds=None` | typed fields on `OutputMetric` | deferred with non-Euclidean output metrics |
| `n_epochs=None` | `Config.Epochs=0` | v1; uses the same size-based 200/500 heuristic and transform epoch rule |
| `learning_rate=1.0` | `Config.LearningRate=1` | v1 |
| `init='spectral'` | `Config.Init=SpectralInit` | v1 spectral and random; PCA, tswspectral, and caller-provided coordinates deferred |
| `min_dist=0.1` | `Config.MinDist=0.1` | v1 |
| `spread=1.0` | `Config.Spread=1` | v1 |
| `low_memory=True` | no option | v1 always uses bounded-memory paths |
| `n_jobs=-1` | `Config.Workers=0` | v1; 0 uses `GOMAXPROCS`; deterministic mode restricts this to one worker |
| `set_op_mix_ratio=1.0` | `Config.SetOpMixRatio=1` | v1 |
| `local_connectivity=1.0` | `Config.LocalConnectivity=1` | v1 |
| `repulsion_strength=1.0` | `Config.RepulsionStrength=1` | v1 |
| `negative_sample_rate=5` | `Config.NegativeSampleRate=5` | v1 |
| `transform_queue_size=4.0` | `Config.Search.TransformQueueSize=4` | v1 |
| `a=None`, `b=None` | `Config.A=0`, `Config.B=0` | v1; both zero derives them; specifying only one is invalid |
| `random_state=None` | `Config.Seed=nil` | v1; nil obtains entropy and records the chosen seed in the model |
| `angular_rp_forest=False` | automatic from metric | v1; implementation detail, no public switch |
| `target_n_neighbors=-1` | `Config.Target.Neighbors=0` | v1; 0 inherits `Neighbors` |
| `target_metric='categorical'` | `Config.Target.Metric=TargetCategorical` | v1 categorical, L1, and L2 |
| `target_metric_kwds=None` | typed fields on `Target.Metric` | v1 for supported target metrics |
| `target_weight=0.5` | `Config.Target.Weight=0.5` | v1 |
| `transform_seed=42` | `Config.TransformSeed=42` | v1 |
| `transform_mode='embedding'` | `Transform` returns embedding | v1; Python graph mode deferred |
| `force_approximation_algorithm=False` | `Config.Search.Mode=SearchAuto` | v1 Auto, Exact, Approximate |
| `verbose=False` | no option | Go library emits no logs; structured progress hooks are deferred |
| `tqdm_kwds=None` | no option | Python UI concern, not applicable |
| `unique=False` | no option | deferred; duplicate handling remains reference-compatible without pre-uniquing |
| `densmap=False` | `Config.Density.Enabled=false` | v1 |
| `dens_lambda=2.0` | `Config.Density.Lambda=2` | v1 |
| `dens_frac=0.3` | `Config.Density.Fraction=0.3` | v1 |
| `dens_var_shift=0.1` | `Config.Density.VarianceShift=0.1` | v1 |
| `output_dens=False` | `Config.Density.Output=false` | v1; radii exposed as immutable model outputs, not a variable-length return tuple |
| `disconnection_distance=None` | `Config.DisconnectionDistance=+Inf` | v1; metric-specific bounded maximum is selected when applicable |
| `precomputed_knn=(None,None,None)` | no v1 constructor option | deferred; importing external search indexes requires a separate ownership/version contract |

Python `fit(X, y)` maps to `Fit(ctx, x, y)`. Python `fit_transform(X, y)`
maps to `FitTransform(ctx, x, y)`, returning both the reusable model and its
embedding. Python `transform(X)` maps to `Model.Transform(ctx, x)` or
`TransformInto`. Go does not mutate a public estimator into a fitted state.

Deferred behavior is: unsupported input/output and callable
metrics, PCA/tswspectral/custom initialization, graph transform mode, external
precomputed neighbors, uniquing, permissive NaN handling, Python object/pickle
compatibility, and progress UI. Model serialization is a separate versioned Go
format and is not Python pickle compatibility.

## Reproducibility

There are two explicit modes:

1. With `Deterministic=true`, a non-nil `Seed`, and one worker, repeated calls
   using identical input bytes, configuration, Go package version, Go toolchain,
   GOARCH, and CPU floating-point features produce bit-identical graph ordering
   and embedding bytes. Ties are ordered by distance then row index. RNG streams
   are derived from the recorded seed and named pipeline stage, never from map
   iteration or goroutine scheduling.
2. With `Deterministic=false`, a fixed seed fixes initialization and sampling,
   but worker scheduling and reductions may change low bits and edge visitation
   order. Results must still satisfy the quality gates below. A nil seed makes no
   repeatability promise, but the selected seed is recorded for replay.

No bitwise promise spans architectures, toolchains, package versions, or changes
to CPU floating-point contraction. Such results use the numerical tolerances.
Transform uses `TransformSeed` and does not advance shared state, so repeated
deterministic transforms are identical and batch size does not affect results.

## Differential correctness gates

All thresholds compare identical validated inputs and effective parameters with
the pinned Python reference. Absolute/relative comparisons use
`abs(go-ref) <= atol + rtol*abs(ref)`. Thresholds are initial release gates; they
may only be relaxed with a fixture, analysis of the divergence, and review.

| Artifact | Required gate |
|---|---|
| Exact kNN distances | identical neighbor indices after deterministic tie-breaking; `atol=1e-6`, `rtol=1e-5` |
| Approximate kNN | recall@k at least `0.95`, and no more than `0.02` below the pinned Python run |
| `rho` and `sigma` | `atol=1e-5`, `rtol=1e-4` |
| Fuzzy graph | identical vertex pairs after zero pruning; weights `atol=1e-6`, `rtol=1e-4` |
| Seeded random initialization | exact values on the same supported platform; otherwise `atol=1e-6`, `rtol=1e-6` |
| Spectral initialization | after orthogonal Procrustes alignment, normalized RMSE at most `0.02` |
| Final training embedding | trustworthiness (k=`Neighbors`) no more than `0.01` below reference; kNN overlap at least `0.90`; Procrustes-aligned RMSE divided by reference RMS radius at most `0.10` |
| Transform embedding | trustworthiness delta at most `0.02`; training-neighbor overlap at least `0.85`; normalized aligned RMSE at most `0.15` |
| Density radii | Pearson correlation at least `0.98` and `atol=1e-4`, `rtol=1e-3` after matching ordering |

Embedding gates are evaluated across dense, sparse, clustered, noisy,
duplicate-heavy, disconnected, tiny, and degenerate fixtures. A final embedding
cannot compensate for an intermediate artifact outside its gate: differential
tests report the first divergent stage. Statistical gates report every fixture,
not only an aggregate mean.
