# Small-input fit overhead

Issue #30 investigated the fixed costs in the matched 64-, 256-, and
1,024-row fit paths. Measurements below were made on Linux/amd64 with Go
1.27.1 on an Intel i5-12600K, using random initialization, 100 epochs, one
worker, and seed 42. End-to-end results use one untimed warmup followed by the
reported measured fits.

## Reproduction

```sh
go test -run '^$' -bench BenchmarkFitStages -benchmem -benchtime=200ms .
go run ./cmd/umap-benchmark \
  --case synthetic/rows/64 --warmups 1 --repeats 7
go run ./cmd/umap-benchmark \
  --case synthetic/rows/1024 --warmups 1 --repeats 5
go test -run '^$' -bench BenchmarkFitStages/rows/64 \
  -benchtime=5x -cpuprofile small-input.cpu.pprof .
go tool pprof -top small-input.cpu.pprof
```

The stage benchmark prepares its inputs outside each timer. Its purpose is to
attribute cost, so its independently timed stages do not sum perfectly to an
end-to-end fit.

## Profile

Representative stage timings after the change:

| Stage | 64 rows | 256 rows |
|---|---:|---:|
| Curve fit, uncached | 29.52 ms | 28.81 ms |
| Curve fit, cached | 16 ns | 16 ns |
| Exact neighbor search | 0.138 ms | 1.99 ms |
| Smooth kNN | 0.130 ms | 0.567 ms |
| Fuzzy graph | 0.213 ms | 1.09 ms |
| Random initialization | 0.0005 ms | 0.0024 ms |
| Layout, 100 epochs | 14.56 ms | 59.38 ms |
| Clone training input | 0.0049 ms | 0.0158 ms |

The dominant fixed cost was curve fitting, not worker setup, exact-search
blocking, graph sorting, context checks, RNG setup, allocation/zeroing, or
model cloning. `FitAB` ran 2,000 optimization steps over 300 samples on every
fit even though its result depends only on `Spread` and `MinDist`.

`FitAB` now caches completed results by the exact IEEE-754 bits of those two
parameters. The first use still performs the same calculation; subsequent
fits reuse the bit-identical result. Concurrent callers use `sync.Map` and may
race only to calculate the same immutable value.

Issue #47 reduced that remaining first-use cost. The default `(spread=1,
min_dist=0.1)` result is now returned as the hexadecimal floating-point values
produced by the original solver, making it bit-identical without repeating an
iterative fit. For other parameter pairs, the 300 target samples and their
logarithms are computed once, outside the 2,000 optimizer steps, and powers are
evaluated from the precomputed logarithms.

On the same Linux/amd64 host, three 500 ms benchmark runs measured the default
cold calculation at about 1.1 ns and a representative non-default pair
`(1.23456789, 0.23456789)` at 5.10--5.14 ms, down from a 26.8--27.1 ms baseline.
Cached lookup remained allocation-free at 15.3--15.5 ns. Run the focused cases
with:

```sh
go test -run '^$' \
  -bench 'BenchmarkFitStages/rows/64/curve-fit' \
  -benchmem -benchtime=500ms -count=3 .
```

`TestFitABRepresentativeGrid` compares the default and varied spread/min-dist
pairs against results captured from the original solver with a `1e-12`
tolerance. The default hexadecimal constants are checked by the same grid, and
the existing concurrent test continues to require bit-identical cache results.

## End-to-end result

| Rows | Before mean | After mean | Change |
|---:|---:|---:|---:|
| 64 | 44.50 ms | 16.72 ms | -62.4% |
| 1,024 | 330.62 ms | 304.63 ms | -7.9% |

The 64-row improvement exceeds the 30% target. The 1,024-row path improves as
well, so it remains within the no-more-than-10% regression requirement. Input
checksums were unchanged (`7a50ad...25b9f` and `92b016...9e00f` respectively),
and the existing deterministic, parity, allocation, and race tests exercise
the same public fit path.
