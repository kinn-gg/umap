.PHONY: test parity-generate parity-full bench-go bench-python
test:
	go test ./...
parity-generate:
	uv run --project parity --locked python parity/generate.py --suite full
parity-full:
	uv run --project parity --locked python parity/generate.py --suite full --output /tmp/umap-parity-full
	PARITY_CORPUS=/tmp/umap-parity-full go test ./internal/parity
bench-go:
	go test -run '^$$' -bench . -benchmem -count 5 ./...
bench-python:
	uv run --project parity --locked python benchmarks/python_reference.py --suite fast --repeats 5
