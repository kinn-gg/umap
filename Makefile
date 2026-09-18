.PHONY: test parity-generate parity-full
test:
	go test ./...
parity-generate:
	uv run --project parity --locked python parity/generate.py --suite full
parity-full:
	uv run --project parity --locked python parity/generate.py --suite full --output /tmp/umap-parity-full
	PARITY_CORPUS=/tmp/umap-parity-full go test ./internal/parity
