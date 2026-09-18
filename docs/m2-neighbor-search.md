# Scalable neighbor search and storage

The graph-building pipeline has two interchangeable k-nearest-neighbor backends.
`SearchExact` is the reference backend: it scans candidate rows in configurable
blocks, retains only the best `k` entries per row, includes the query row as a
self-neighbor first, and resolves remaining equal distances by row index. `SearchNNDescent`
uses seeded sampling and deterministic neighbor propagation. `SearchAuto`
selects exact search through 4,096 rows and NN-descent above that threshold.

```go
cfg := umap.DefaultConfig()
cfg.NeighborSearch.Algorithm = umap.SearchNNDescent
cfg.NeighborSearch.Approximate = umap.NNDescentOptions{
    MaxIterations:     12,
    CandidatePoolSize: 64,
    ConvergenceDelta:  0.001,
}
cfg.MemoryBudget = 64 << 20
cfg.Progress = func(p umap.SearchProgress) { /* safe iteration boundary */ }
```

Cancellation is checked after every exact block and before every NN-descent
iteration. A fixed UMAP seed fixes approximate-search output. Dense rows use
contiguous kernels; CSR/CSR and CSR/dense kernels merge sorted column indices
without allocating or densifying. Correlation accounts for implicit zeros.

## Memory model

`EstimateGraphMemory` predicts retained bytes from rows (`n`), neighbors (`k`),
input dimensions or CSR nonzeros, worker count, algorithm, and exact block size:

- dense input: `4*n*dimensions`; CSR input: `8*nnz + 8*(n+1)`;
- neighbor output: `n*k*(sizeof(int)+4)`;
- fuzzy graph storage: at most `2*n*k*sizeof(Edge)`;
- exact temporary state: `8*workers*blockSize`;
- NN-descent temporary state: the `n*k` index snapshot, a four-byte-per-row
  initialization stamp array, plus `workers*k*k` candidate entries.

The input term describes caller-owned storage and lets peak-RSS tests account
for the whole process. Exact `MemoryBudget` chooses its block size; workers are
currently one because Milestone 2 deliberately preserves single-threaded
reproducibility. Fuzzy symmetrization sorts compact edge slices rather than
building hash maps, so retained graph memory remains O(n*k).

## Quality and crossover measurements

`TestNNDescentReproducibleRecallProgressAndCancellation` enforces Recall@10 >=
0.90 on a fixed 240-by-8 corpus. `BenchmarkNeighborSearchCrossover` compares
both backends over increasing row counts. `BenchmarkNNDescentTextLike99PercentSparse`
uses a 1,000-by-10,000 CSR corpus with 0.5% density. Run them with:

```sh
go test -run '^$' -bench 'NeighborSearchCrossover|TextLike' -benchmem
```

The exact/approximate crossover is hardware- and metric-dependent; the
benchmark is the source of truth when tuning `SearchAuto` for a deployment.
