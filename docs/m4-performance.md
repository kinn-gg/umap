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
