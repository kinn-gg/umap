# Differential parity harness

This directory compares staged Go pipeline output with `umap-learn` at commit
`cdb0eeb236f5d6759e0c3dd1fe132aef4db9e7b1` (release 0.5.12). JSON artifacts
contain the input, effective parameters, seed, checksum, runtime/package
versions, exact kNN output, `rho`/`sigma`, fuzzy graph, initialization, and final
embedding. The six deterministic generators cover dense, sparse, clustered,
noisy, duplicated, and degenerate inputs.

Regenerate the corpus with
[`uv`](https://docs.astral.sh/uv/). The committed lockfile supplies Python
3.11.16 and the complete pinned dependency graph automatically:

```sh
uv run --project parity --locked python parity/generate.py --suite full
```

Compare a Go-produced artifact with its reference:

```sh
go run ./cmd/umap-parity -reference parity/fixtures/dense.json -candidate candidate.json
```

The command exits nonzero and names the first divergent stage (`input`, `knn`,
`smooth_knn`, `fuzzy_graph`, `initialization`, or `embedding`). Embeddings are
judged using trustworthiness, neighbor overlap, pairwise stress, and centered
Procrustes-aligned normalized RMSE, so rotation and reflection do not cause
false failures.

Thresholds come directly from `docs/v1-contract.md`: exact-neighbor distances
use `atol=1e-6, rtol=1e-5`; local-connectivity values use `1e-5, 1e-4`; graph
weights use `1e-6, 1e-4`; spectral initialization uses aligned NRMSE `0.02`;
and final embeddings allow trustworthiness delta `0.01`, overlap `0.90`, and
aligned NRMSE `0.10`. These scale-aware gates avoid bitwise floating-point
assumptions across supported platforms.

`go test ./...` validates the checked-in fast subset without requiring Python.
`make parity-full` uses the locked uv environment to generate and validate all
six fixtures separately under `/tmp`; `make parity-generate` intentionally
refreshes the versioned corpus.
