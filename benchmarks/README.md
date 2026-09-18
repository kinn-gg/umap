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
go test -run '^$' -bench . -benchmem -count 5 -json ./internal/parity \
  > go-bench.jsonl
```

`BenchmarkCompareScaling` varies one synthetic axis at a time: rows,
dimensions, neighbors, components, graph density, and metric. It reports
wall-time, allocations/op, and allocated bytes/op. `BenchmarkCorpusEndToEnd`
includes artifact decoding, validation, and full staged comparison for the
checked-in corpus. These currently establish the overhead of the differential
harness; pipeline-stage benchmarks should use the same axis names and dataset
profiles as Go stages land.

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
