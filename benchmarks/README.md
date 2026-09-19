# Performance and memory benchmarks

The benchmark suite measures runtime, allocations, retained heap, and process
peak RSS. Correctness is covered separately by the differential harness in
[`parity/`](../parity/README.md). Machine-readable results include tool
versions, platform and hardware details, warmup policy, parameters, and an
SHA-256 checksum of every input.

The full suite is intentionally local-only because it is too slow for routine
CI. Pull requests run correctness, race, parity, and focused allocation checks.

## Go benchmarks

Run all Go benchmarks with machine-readable output:

```sh
go test -run '^$' -bench . -benchmem -count 5 -json ./... \
  > go-bench.jsonl
```

Useful focused benchmarks include:

- `BenchmarkCompareScaling` for rows, dimensions, neighbors, components, graph
  density, and metric scaling;
- `BenchmarkFitStages` for curve fitting, neighbor search, smooth-kNN, graph
  construction, initialization, layout, and model-input cloning;
- `BenchmarkNeighborSearchCrossover` for exact versus approximate search;
- `BenchmarkLayoutOptimization` for layout dimensions and graph density;
- `BenchmarkSparseTextFitStages` for sparse cosine workloads; and
- `BenchmarkDensityOverhead` for densMAP overhead.

For example:

```sh
go test -run '^$' -bench BenchmarkFitStages -benchmem -count 5 .
go test -run '^$' -bench BenchmarkLayoutOptimization -benchmem -count 5 .
make bench-density
```

## Profiles and peak RSS

The Makefile provides reproducible CPU and heap profile targets:

```sh
make profiles
make profiles-small
make profiles-sparse
make profiles-low-density
```

Inspect profiles with `go tool pprof`. Measure peak RSS for any Go benchmark or
Python reference invocation with the wrapper:

```sh
go run ./cmd/peakrss --output rss.json -- \
  go test -run '^$' -bench BenchmarkCorpusEndToEnd -benchtime 5x ./internal/parity
```

The wrapper records wall time and child-process high-water RSS on Linux and
macOS. Windows records runtime and exit status, but peak RSS is `null`.

## Python reference

The reference runner uses the locked environment in `parity/`:

```sh
uv run --project parity --locked python benchmarks/python_reference.py \
  --suite fast --repeats 5 --output python-bench.jsonl
```

The fast suite covers synthetic scaling cases plus Iris, Digits, and sparse
text-like data. The full suite additionally uses deterministic MNIST and
Fashion-MNIST subsets. Prefetch public datasets with `--prepare-public` so
downloads are excluded from measurements. Use `--warmups 0` when JIT startup
time is part of the workload being measured.

## Matched Go/Python comparison

Run the end-to-end comparison with:

```sh
make bench-compare
```

The suite fits both implementations on byte-identical dense and CSR inputs
with matching seeds, metrics, neighbor counts, component counts, epoch counts,
initialization, and worker count. It writes `go-umap-bench.jsonl`,
`python-umap-bench.jsonl`, and `umap-comparison.json`; these generated files are
ignored by Git.

## Regression policy

Compare only results with matching inputs, architecture, tool versions, and
warmup policy. A median runtime or peak-RSS increase above 10%, or any nonzero
allocation in a metric inner loop, is a regression. Rerun once to exclude
transient noise. If the second run still exceeds the budget, include the
benchmark artifacts and an explicit rationale in the pull request.
