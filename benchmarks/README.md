# Reproducible performance and memory benchmarks

The benchmark suite measures the three project priorities independently:
correctness is handled by `parity/`, while this directory records runtime,
allocations, retained heap, and process peak RSS. Results are JSON or JSONL and
include tool versions, platform, hardware, warmup policy, parameters, and an
SHA-256 checksum of every input.

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
The `curve-fit/cold` and `curve-fit/cached` stages keep both costs visible.

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
`umap-comparison.json`. CI also stores a Markdown summary. Go allocated bytes
come from `runtime.MemStats.TotalAlloc`; Python's allocation peak comes from
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
