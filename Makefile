.PHONY: test race parity-generate parity-full bench-go bench-density bench-python bench-compare profiles profiles-sparse
test:
	go test ./...
race:
	go test -race ./...
parity-generate:
	uv run --project parity --locked python parity/generate.py --suite full
parity-full:
	uv run --project parity --locked python parity/generate.py --suite full --output /tmp/umap-parity-full
	PARITY_CORPUS=/tmp/umap-parity-full go test ./internal/parity
bench-go:
	go test -run '^$$' -bench . -benchmem -count 5 ./...
bench-density:
	go test -run '^$$' -bench BenchmarkDensityOverhead -benchmem -count 5 .
bench-python:
	uv run --project parity --locked python benchmarks/python_reference.py --suite fast --repeats 5
bench-compare:
	go run ./cmd/umap-benchmark --repeats 5 --output go-umap-bench.jsonl
	uv run --project parity --locked python benchmarks/python_reference.py --suite matched --repeats 5 --output python-umap-bench.jsonl
	uv run --project parity --locked python benchmarks/compare.py --go go-umap-bench.jsonl --python python-umap-bench.jsonl --output umap-comparison.json
profiles:
	go test -run '^$$' -bench BenchmarkExactNeighborWorkers -benchtime 5x -cpuprofile cpu.pprof -memprofile heap.pprof .
profiles-sparse:
	go test -run '^$$' -bench BenchmarkExactSparseTextCosine -benchtime 3x -cpuprofile sparse-cpu.pprof -memprofile sparse-heap.pprof .
