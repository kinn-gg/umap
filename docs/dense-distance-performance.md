# Dense distance performance

Issue #34 investigated distance work in the matched 256-row,
128-dimensional fit. Measurements were made on Linux/amd64 with Go 1.27.1
on an Intel i5-12600K. Each microbenchmark below used five runs and reported
zero allocations per distance call.

## Implementation

Dense Euclidean distance now dispatches once per pair to a portable,
four-value unrolled kernel instead of switching on the metric for every
coordinate. The accumulation order is unchanged. Dense neighbor search also
captures the concrete matrix once, avoiding repeated interface type switches
and row-shape checks.

Dense cosine neighbor search precomputes one squared norm per row and only
calculates the dot product for each pair. The cached values are immutable and
fit-scoped, so concurrent search remains safe. The public `Metric.Distance`
cosine path retains its single-pass implementation because separate norm
passes regressed wider one-off calls.

## Results

Representative median Euclidean kernel results:

| Dimensions | Before | After | Change |
|---:|---:|---:|---:|
| 2 | 3.80 ns | 3.13 ns | -17.6% |
| 16 | 18.80 ns | 9.86 ns | -47.6% |
| 128 | 121.9 ns | 77.7 ns | -36.3% |
| 512 | 462.2 ns | 310.9 ns | -32.7% |

The existing 256-row, 16-dimensional exact-neighbor benchmark improved from
approximately 475-506 microseconds to 402-413 microseconds. The matched
256-row, 128-dimensional end-to-end case averaged 72.87 ms across seven runs,
down 20.1% from the issue baseline of 91.25 ms and exceeding the 15% target.
Its input checksum remained
`192de6cc60817f3be1407273660aee9aca43488cbd5a09dd292755559c689a06`.

## Reproduction

```sh
go test -run '^$' -bench BenchmarkDenseDistanceDimensions -benchmem -count 5 .
go test -run '^$' -bench BenchmarkDenseFitDimensions -benchmem -benchtime=3x .
go run ./cmd/umap-benchmark \
  --case synthetic/dimensions/128 --warmups 1 --repeats 7
```
