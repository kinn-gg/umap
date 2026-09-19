# M4 performance and memory hardening

Neighbor search uses bounded row-parallel worker pools. `Config.Workers` sets
the cap; zero uses `GOMAXPROCS`, and the effective count never exceeds the row
count. Exact search further reduces the count when its explicit memory budget
cannot cover per-worker block scratch. Each row owns a disjoint heap segment,
so metric loops need no locks or logically shared writes.

NN-descent derives one RNG stream per row from the configured seed. Scheduling
therefore cannot alter candidate selection. Exact and approximate search are
reproducible across worker counts; `go test -race ./...` covers both paths.

## Profiling and scaling

Run `make profiles` to capture CPU, heap, and allocation profiles for the
worker-scaling workload. Run this command for comparable scaling curves:

```sh
go test -run '^$' -bench BenchmarkExactNeighborWorkers -benchmem -benchtime 5x .
```

Record the Go version, OS, architecture, CPU model, commit, and worker count
beside results. `SmoothKNN` now calculates its global distance mean once rather
than once per row, removing an O(rows² × neighbors) scan. Built-in metric loops
remain zero-allocation and are protected by `TestHotPathAllocationGate`.

Baseline captured on Linux/amd64, Go 1.24, i5-12600K, 512 × 32 dense input,
15 neighbors, three iterations:

| Workers | ns/op | bytes/op |
|---:|---:|---:|
| 1 | 14,840,222 | 98,544 |
| 2 | 8,812,637 | 99,018 |
| 4 | 5,290,816 | 100,768 |
| 8 | 3,446,909 | 105,130 |
| 16 | 2,532,678 | 110,416 |

Eight workers provide 4.3× throughput with 6.7% allocated-byte growth. The
worker cap and memory estimate prevent unbounded per-worker growth.

## Regression policy

Run `make bench-density` to measure densMAP separately. Its `disabled` and
`enabled` sub-benchmarks use identical data, seeds, and epochs; the difference
is the density objective and its `O(rows)` radius/error buffers. The disabled
case is also the ordinary-path guard: density radius buffers are never created
unless densMAP or density output is requested.

Every pull request runs parity, the race detector, supported Go versions, the
allocation gate, five-sample Go and pinned-Python benchmarks, and peak RSS.
The benchmark workflow stores its machine-readable artifacts for 90 days.
Compare only artifacts with
matching inputs, architecture, Go/Python versions, and warmup policy.

A median runtime or peak-RSS increase above 10%, or any nonzero metric-loop
allocation, is a regression. Rerun the job once to reject transient noise. If
the second run also exceeds the budget, the change must fail review unless its
PR includes benchmark artifacts and an explicit baseline-update rationale.
Budgets may never be relaxed merely to make a job pass.

## Smooth-kNN calibration (#33)

`SmoothKNN` finds the local-connectivity samples directly in each ordered
distance row instead of copying all positive distances into a temporary slice.
It also prepares the float64, rho-adjusted distances once per row for reuse by
the binary search. Neighbor counts up to 65 use stack scratch; larger counts
share one function-scoped fallback slice. Thus allocation count is independent
of the number of rows.

Use the focused benchmark below for distinct and duplicated-distance inputs at
64, 256, and 1,024 rows:

```sh
go test -run '^$' -bench '^BenchmarkSmoothKNNCalibration$' -benchmem -count 5 .
```

On Linux/amd64, Go 1.27.1, i5-12600K, five-sample medians changed as follows:

| Input | Baseline | Optimized | Change | Allocations |
|---|---:|---:|---:|---:|
| 256 distinct | 517 us | 473 us | -8.5% | 258 to 2 |
| 256 duplicated | 472 us | 445 us | -5.7% | 258 to 2 |
| 1,024 distinct | 2.051 ms | 1.890 ms | -7.9% | 1,026 to 2 |
| 1,024 duplicated | 1.887 ms | 1.778 ms | -5.8% | 1,026 to 2 |

The 15% runtime target is not reached consistently. After removing row
allocation and repeated distance conversion, the remaining loop is primarily
the required `math.Exp` evaluation for every neighbor and calibration step.
Reducing their count or replacing each division with reciprocal multiplication
changes floating-point evaluation and can change rho/sigma convergence. The
optimization therefore retains the existing 64-iteration bound, tolerance,
and arithmetic while eliminating more than 99% of allocations and reducing
allocated bytes by 88.9%.

## Fuzzy graph construction (#37)

`BenchmarkFuzzyGraph` measures graph construction for 64, 256, and 1,024 rows
at both 15 and 50 neighbors. The optimized path records row offsets after the
directed memberships are sorted, limits each reverse-edge binary search to one
row, and performs one lookup per membership. Final edges are grouped in linear
time and only the tails within each row are sorted. The original first-entry
rule for duplicate directed memberships and the final `(head, tail)` ordering
remain bit-identical.

Five local runs on Linux/amd64, Go 1.27.1, i5-12600K produced these medians:

| rows | neighbors | before | after | change |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 15 | 0.227 ms | 0.106 ms | -53.1% |
| 64 | 50 | 0.900 ms | 0.609 ms | -32.3% |
| 256 | 15 | 1.120 ms | 0.651 ms | -41.8% |
| 256 | 50 | 4.247 ms | 2.526 ms | -40.5% |
| 1,024 | 15 | 4.621 ms | 2.457 ms | -46.8% |
| 1,024 | 50 | 17.451 ms | 7.693 ms | -55.9% |

At the acceptance case (256 rows, 50 neighbors), allocated bytes increased
from 917,696 to 921,952 (+0.46%) while allocations fell from eight to seven.
The two row-sized integer work arrays account for the small increase; retained
graph storage is unchanged. A representative 256-row, 16-dimensional
end-to-end fit moved from a 43.6 ms median to 43.1 ms (-1.2%), so the matched
pipeline stays within the 5% regression threshold. Reproduce the graph results
with:

```sh
go test -run '^$' -bench '^BenchmarkFuzzyGraph$' -benchmem -count 5 .
```

## Sparse cosine investigation (#29)

The matched `synthetic/sparse-text` workload was dominated by exact neighbor
search. Auto selection chose exact search because 1,000 rows is below its
4,096-row crossover, and the original path evaluated the full 1,000 x 1,000
distance matrix. Every distance also recomputed the squared norms of both CSR
rows. The cost therefore came from repeated sparse cosine work, rather than
temporary allocation or densification.

Sparse cosine search now prepares one `float64` squared norm per row for the
duration of the search. Single-worker exact search also visits one triangular
half of the symmetric distance matrix and offers each result to both neighbor
rows. Its iteration order preserves the full scan's ascending candidate order,
including deterministic distance ties. Multi-worker exact search retains the
row-owned full scan so it needs no locks or per-worker graph-sized buffers.

Reproduce the isolated stage benchmark and profiles with:

```sh
go test -run '^$' -bench '^BenchmarkExactSparseTextCosine$' -benchmem -benchtime 5x .
make profiles-sparse
go tool pprof -top sparse-cpu.pprof
```

On Linux/amd64, Go 1.27.1, i5-12600K, one worker, the isolated optimized exact
search takes about 62 ms and allocates 197 KiB. Five matched end-to-end samples
improved from a local 791 ms baseline to 442 ms. Compared with the issue's
1,123 ms Go artifact, the result is 2.5x faster. Allocated bytes changed from
1.933 MiB to 1.941 MiB (+0.4%) and peak RSS from 25.1 MB to 26.4 MB (+5.1%),
both inside the 10% memory budget. The remaining end-to-end time is primarily
layout optimization; neighbor search is no longer the limiting stage for this
case.

## Sparse cosine backend selection (#46)

Automatic exact search now chooses between the single-worker inverted index
and the parallel full scan with a posting-aware cost model. Index work is
estimated as `rows^2 + sum(column_frequency^2)`: the first term represents the
candidate pass required to preserve zero-dot ties, and the second represents
dot-product accumulation through posting lists. Full-scan work is estimated as
`rows^2 * 2 * mean_row_nonzeros / workers`. A calibrated factor of four charges
the index for scattered posting reads and candidate bookkeeping. Explicit
`Workers` values are unchanged: one selects the index when eligible and larger
values select the corresponding parallel full scan. A `MemoryBudget` continues
to disable the index and bound the full-scan block and worker counts.

The crossover matrix is reproducible with:

```sh
go test -run '^$' -bench '^BenchmarkExactSparseCosineCrossover$' \
  -benchmem -benchtime 5x -count 3 .
```

On Linux/amd64, Go 1.27.1, i5-12600K, the 512 x 1,024 input with eight
nonzeros per row moved from about 6.8 ms on the index to 1.6 ms automatically
(the 8-worker full scan took 2.5 ms). The uniform 1,000 x 5,000 input with 50
nonzeros per row retained the index at about 25.2 ms, statistically level with
the fastest explicit eligible backend in this run. A maximally skewed version
correctly selected the full scan (about 15.3 ms versus 153 ms for the index).
The existing matched end-to-end sparse-text benchmark remained about 208 ms,
inside its 5% regression allowance. Results vary with CPU scheduling; the
automatic path can use all `GOMAXPROCS` workers while the table's explicit
comparisons use four and eight.
