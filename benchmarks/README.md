# Reproducible performance and memory benchmarks

The benchmark suite measures the three project priorities independently:
correctness is handled by `parity/`, while this directory records runtime,
allocations, retained heap, and process peak RSS. Results are JSON or JSONL and
include tool versions, platform, hardware, warmup policy, parameters, and an
SHA-256 checksum of every input.

The full benchmark suite is intentionally local-only because it is too slow
for routine CI. Pull requests still run correctness, race, parity, and focused
allocation regression tests; run the commands below when evaluating
performance changes.

## Go measurements

Run the micro and end-to-end comparison benchmarks with Go's machine-readable
output enabled:

```sh
go test -run '^$' -bench . -benchmem -count 5 -json ./... \
  > go-bench.jsonl
```

`BenchmarkCompareScaling` varies one synthetic axis at a time: rows,
dimensions, neighbors, components, graph density, and metric. It reports
wall-time, allocations/op, and allocated bytes/op. `BenchmarkCorpusEndToEnd`
includes artifact decoding, validation, and full staged comparison for the
checked-in corpus. These currently establish the overhead of the differential
harness; pipeline-stage benchmarks should use the same axis names and dataset
profiles as Go stages land.

The root package includes `BenchmarkNeighborSearchCrossover`, which compares
blocked exact search with NN-descent at 256, 1,024, and 4,096 rows, and
`BenchmarkNNDescentTextLike99PercentSparse`, which exercises 0.5%-dense CSR
data. Recall and fixed-seed reproducibility are enforced separately in the
fast test suite so benchmark tuning cannot silently trade away quality.

`BenchmarkFitStages` profiles the matched 64- and 256-row configurations by
curve fitting, exact neighbor search, smooth-kNN calibration, fuzzy-graph
construction, random initialization, layout optimization, and model-input
cloning. Run it independently to avoid mixing stage timings with the parity
harness:

```sh
go test -run '^$' -bench BenchmarkFitStages -benchmem -count 5 .
```

The small-input investigation found that the input-independent `FitAB` curve
fit dominated the 64-row path: it evaluates 300 samples for 2,000 optimizer
steps for every call. Completed fits are now cached by the exact IEEE-754 bits
of `(spread, min_dist)`. This preserves the fitted values and concurrency
safety while removing that fixed cost after the first use of a configuration.
The default pair also has a bit-identical precomputed result. Non-default fits
precompute the target samples and logarithms used by every optimizer step.
The `curve-fit/cold/default`, `curve-fit/cold/non-default`, and
`curve-fit/cached` stages keep all three costs visible.

### Layout optimization profile

`BenchmarkLayoutOptimization` isolates the 100-epoch layout cases from issues
#35 and #45, including two- and eight-component embeddings at 15 and 50
neighbors. On an Intel i5-12600K with Go 1.27.1, three-run medians for the
original two-component work were:

| Rows | Neighbors | Before | After | Change |
|---:|---:|---:|---:|---:|
| 64 | 15 | 14.59 ms | 10.33 ms | -29.2% |
| 256 | 15 | 58.35 ms | 41.36 ms | -29.1% |
| 256 | 50 | 86.71 ms | 61.91 ms | -28.6% |
| 1,024 | 15 | 237.45 ms | 165.60 ms | -30.3% |

The matched 256-row/50-neighbor end-to-end case fell from a 106.93 ms mean
to 79.89 ms across five measured runs (-25.3%). Allocations remain at five
per layout call with unchanged allocated bytes.

Before the change, a three-second CPU profile of the 256-row/15-neighbor case
attributed 59.6% of samples directly to `math.archExp`, `math.archLog`, and
`math.pow`; `math.pow` was cumulatively responsible for 73.5%. Afterward,
`fastPowf` accounts for 72.7% and the standard transcendental routines for
less than 1%, making the remaining cost explicit rather than hiding it behind
`math.Pow` dispatch and duplicate exponentiation.

The optimized layout loops compute each distance power once using float32
range reduction and bounded polynomials; `TestFastPowfAccuracy` caps relative
error at 2e-5 over the exercised exponent range. A pre-filtered edge schedule
was also measured, but rejected: it improved the
256-row/50-neighbor case by only about 4% while increasing allocated bytes by
roughly 2.6x. Keeping the existing edge scan preserves its compact allocation
budget and deterministic update order.

Issue #52 specializes the power evaluator for the fixed exponent used by each
layout call. A 257-entry mantissa table and 254-entry exponent table are built
once on the stack, so every sampled update needs only one linear interpolation
and one multiplication instead of separate interpolated log2 and exp2 passes.
Unusual and out-of-range float32 values retain the standard-library fallback.
On the same i5-12600K, local three-run medians with Go 1.27.1 were:

| Rows | Neighbors | Components | Before | After | Change |
|---:|---:|---:|---:|---:|---:|
| 64 | 15 | 2 | 6.58 ms | 3.65 ms | -44.5% |
| 256 | 15 | 2 | 26.35 ms | 14.68 ms | -44.3% |
| 256 | 50 | 2 | 39.24 ms | 22.23 ms | -43.3% |
| 256 | 15 | 8 | 29.63 ms | 19.22 ms | -35.1% |
| 256 | 50 | 8 | 45.01 ms | 29.69 ms | -34.0% |
| 1,024 | 15 | 2 | 105.24 ms | 60.00 ms | -43.0% |

The matched sparse-text layout median fell from 110.69 ms to 61.63 ms
(-44.3%). Every layout case remains at five allocations with unchanged bytes.
In a post-change end-to-end CPU profile, the evaluator accounts for 20.5% of
samples flat and the optimizer for 54.6% cumulatively, down from the issue's
58% and 79% respectively. The three-run end-to-end benchmark measured
99.79 ms/op and 36 allocations.

For issue #45, a CPU profile of the eight-component, 256-row/50-neighbor case
showed standard `math.Pow` consuming 60.5% of samples cumulatively in the
negative-sampling path even after attractive updates used `fastPowf`. Applying
the same single-power formulation to both general-component updates removes
the float64 exponentiation without changing update or RNG order. Five-run
medians were:

| Rows | Neighbors | Components | Before | After | Change |
|---:|---:|---:|---:|---:|---:|
| 256 | 15 | 8 | 61.12 ms | 44.49 ms | -27.2% |
| 256 | 50 | 8 | 90.54 ms | 66.49 ms | -26.6% |

Both cases remain at five allocations per call, with allocated bytes unchanged
at 57,360 and 180,240 respectively. In the optimized 50-neighbor profile,
`fastPowf` accounts for 62.7% cumulatively and standard transcendental power
functions disappear from the sampled layout path. The density path benefits
from the same calculation when enabled; its existing per-epoch radii work is
otherwise unchanged.

Reproduce the layout measurements and profile with:

```sh
go test -run '^$' -bench BenchmarkLayoutOptimization -benchmem -count 3 .
go test -run '^$' -bench 'BenchmarkFitStages/rows/256/layout$' \
  -benchtime=3s -cpuprofile=layout.pprof .
go tool pprof -top layout.pprof
```

### Sparse-text fit profile

`BenchmarkSparseTextFitStages` uses the matched 1,000-row by 5,000-dimension,
1%-dense cosine input and separates CSR validation, input cloning, exact
neighbors, smooth-kNN, graph construction, random initialization, and layout.
`BenchmarkSparseTextFitEndToEnd` is the corresponding profile target:

```sh
go test -run '^$' -bench BenchmarkSparseTextFitStages -benchmem -count 3 .
make profiles-sparse
go tool pprof -top sparse-cpu.pprof
go tool pprof -top -alloc_space sparse-heap.pprof
```

On an Intel i5-12600K with Go 1.27.1, the median isolated neighbor-search
time fell from 57.7 ms to 24.8 ms (-57.0%). The optimized search builds a
compact column-to-row index once, accumulates only non-zero sparse dot
products, and still offers every candidate in ascending row order. It is
therefore exact and preserves zero-dot ties, neighbor recall, and seeded
repeatability. Searches with an explicit memory budget retain the bounded
full-scan path so the index cannot bypass the caller's limit.

The matched five-run end-to-end mean fell from 362.7 ms to 210.0 ms (-42.1%).
This is 61.4% below the 544.4 ms starting point recorded in issue #36. Stage
medians after the change were 0.07 ms validation, 0.44 ms cloning, 31.8 ms
neighbors, 2.39 ms smooth-kNN, 4.75 ms graph construction, 0.01 ms random
initialization, and 172.2 ms layout. Layout is the dominant stage at about
81% of the summed stage time; exact neighbors account for about 15%.

The inverted index adds about 504 KiB of temporary allocation to the isolated
search. In the matched process, retained Go heap stayed effectively flat
(1.111 MB before and after) and peak RSS moved from 25.6 MB to 26.1 MB. Exact
search remains the automatic choice at this shape: approximate search would
trade away exact recall and repeatability without addressing the dominant
layout stage. Dense inputs do not enter this specialization.

Measure peak RSS for any Go benchmark or Python reference invocation:

```sh
go run ./cmd/peakrss --output rss.json -- \
  go test -run '^$' -bench BenchmarkCorpusEndToEnd -benchtime 5x ./internal/parity
```

The wrapper records wall time and the child process high-water RSS on Linux and
macOS. Windows still records runtime and exit status, but peak RSS is `null`;
portable access to `PeakWorkingSetSize` will be added when the project accepts
a Windows system-call dependency.

## Pinned Python reference

The reference runner uses the same locked environment as the parity fixtures:

```sh
uv run --project parity --locked python benchmarks/python_reference.py \
  --suite fast --repeats 5 --output python-bench.jsonl
```

The fast suite covers every synthetic scaling axis plus Iris, Digits, and
sparse text-like data. `--suite full` additionally uses deterministic MNIST and
Fashion-MNIST subsets through OpenML. Network downloads are never part of
timing; prefetch them with `--prepare-public` to keep benchmark runs offline.
Each measured case performs one untimed warmup by default, which means
Numba JIT cost is explicitly excluded and reported as `warmup_ns`. Pass
`--warmups 0` to capture cold/JIT-inclusive behavior. Python allocation and
retained-heap values come from a separate `tracemalloc` probe, so tracing
overhead does not contaminate the reported runtime samples.

## Matched Go/Python comparison

Run the end-to-end comparison with:

```sh
make bench-compare
```

The matched suite fits both implementations on byte-identical dense and CSR
inputs with the same seed, metric, neighbor count, component count, 100 epochs,
random initialization, and one worker. A small cross-language SplitMix64 input
generator avoids checking large benchmark datasets into the repository. The
comparison command verifies every parameter object and canonical input SHA-256
before reporting median fit time and allocation measurements; it fails rather
than comparing mismatched cases.

The generated files are `go-umap-bench.jsonl`, `python-umap-bench.jsonl`, and
`umap-comparison.json`. Go allocated bytes come from
`runtime.MemStats.TotalAlloc`; Python's allocation peak comes from
`tracemalloc`, so each is useful for regressions within its runtime but their
ratio is not a direct heap-efficiency claim. Peak RSS includes each language
runtime and rises monotonically over the process; compare matching case order
and runner versions only.

## Initial budgets

These are infrastructure budgets, not claims about the future Go UMAP runtime.
They are deliberately conservative and must not be relaxed without attaching
machine-readable results and explaining the regression.

| Measurement | Initial budget |
|---|---:|
| Checked-in fast Go tests | under 10 seconds |
| Parity comparator, 1,024 rows / 15 neighbors | under 2 seconds/op |
| Parity comparator allocations, 1,024 rows | under 256 MiB/op |
| Peak-RSS wrapper overhead | under 50 MiB retained RSS |
| Python reference variability after warmup | coefficient of variation under 15% across 5 runs |

Historical comparisons must match dataset checksum, effective parameters,
warmup count, suite version, architecture, and runtime versions. The Python
runner records warmup and measured time separately so JIT cost is never hidden.

## Canonical matched result after sparse-selector retuning

The checked-in artifacts were generated from main commit
`ca3081c74094e4b406f317b3711ff7b949909713` with the standard one untimed
warmup and five measured repeats. The host was Linux 7.0.0-31-generic on a
16-logical-CPU Intel Core i5-12600K. The toolchain was Go 1.27.1, uv 0.12.7,
Python 3.11.16, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.6.1, Numba 0.61.2,
PyNNDescent 0.5.13, and umap-learn 0.5.12.

All 13 matched cases were faster in Go. Python/Go median runtime ratios ranged
from 1.72x for `synthetic/sparse-text` to 4.74x for `synthetic/rows/64`. The
largest Go coefficient of variation was 2.03% (`synthetic/dimensions/128`),
and the largest Python coefficient of variation was 8.74%
(`synthetic/metric/euclidean`). Go peak RSS was 24.6 MiB throughout the suite.

The canonical input SHA-256 values are:

| Cases | SHA-256 |
|---|---|
| `synthetic/rows/64` | `7a50ad82588bb9edda9dcce1281705de3ac95c555f07204209d57cca5cb25b9f` |
| `synthetic/rows/1024` | `92b016032eef36b8889ce69d27243a5d6c778172392240a33e0d3d4f2c89e00f` |
| `synthetic/dimensions/2` | `b0a7bc92803cc31d9a810186eb2e23cbc860159216079a95fd2d15b437489eb7` |
| `synthetic/dimensions/128` | `192de6cc60817f3be1407273660aee9aca43488cbd5a09dd292755559c689a06` |
| `synthetic/neighbors/{5,50}`, `synthetic/components/{2,8}`, `synthetic/metric/{euclidean,cosine}` | `b73ee4d65e1a52d868d2b39a862cfa63e40537c0245af8d0c50902e0cbbcb1c5` |
| `synthetic/density/0.05` | `b5c2b256959885a3ea2aae1794d80f95ddd62541672bfc1b6fc34dc4a0d89c06` |
| `synthetic/density/0.5` | `cdd74ad6b68281319d82032a7b42ef19fb7b524ff4e2d14b3b95fbf98150b15b` |
| `synthetic/sparse-text` | `30d311188c145ce22fbe0e69f28977394a9bed0a42a1ac7eb71393aab0bc0c95` |

The JSONL files retain every measured sample, mean, standard deviation,
allocation probe, and peak-RSS observation. `umap-comparison.json` is the
validated, machine-readable summary.

## 10k-100k scaling suite

The local-only scaling suite exercises dense 64-dimensional Euclidean and
sparse 4,096-dimensional cosine inputs at 10,000, 50,000, and 100,000 rows.
Every case uses 15 neighbors, two output components, 100 layout epochs, seed
42, and the explicitly selected NN-descent backend with a 256-candidate pool,
20 maximum iterations, and 0.0001 convergence delta. Sparse rows contain 16
sorted nonzeros (density 0.00390625). Run all six cases with:

```sh
make bench-scaling
```

The command intentionally has no routine CI target. It performs one warmup and
three measured fits by default, and runs each workload in a fresh child process
so its peak RSS is independent of earlier cases. Linux and macOS report the
process high-water RSS; other platforms emit `null` and the metadata identifies
that resource limit. Use `--case dense/10000` (or another exact case name) for
a focused run. `--workers`, `--warmups`, `--repeats`, `--epochs`, and
`--recall-queries` make resource use explicit; at least two repeats are
required.

The JSONL begins with a metadata record containing the suite version, runtime,
OS, architecture, CPU, logical CPU count, module/VCS versions, seed, warmup and
repeat counts, effective workers, epochs, recall sample size, RSS support, and
budget policy. Each case then records all effective parameters, selected
backend, canonical input SHA-256, every warmup and measured end-to-end,
neighbor-search, and layout duration, an isolated allocation probe, retained
Go heap, peak RSS, and quality results.

Recall@15 is measured against exhaustive full-corpus neighbors for evenly
spaced, fixed query rows (32 by default), rather than against a smaller corpus.
NN-descent is run twice with seed 42 and its complete indices and distances
must match for `seeded_reproducible` to be true. This quality probe is outside
the timed fit samples.

Initial per-machine regression limits are derived independently for total,
neighbor-search, and layout time from the repeated measurements. The limit is
the median plus the larger of six median absolute deviations or 25% of the
median. Never compare runs unless the input checksum, suite version, runtime,
architecture, worker count, epochs, warmup count, seed, and parameters match.
Treat the generated limits as a baseline for that host, not as portable
performance claims; preserve the baseline JSONL with any regression report.
