.PHONY: test parity-generate parity-full
test:
	go test ./...
parity-generate:
	python3 parity/generate.py --suite full
parity-full:
	python3 parity/generate.py --suite full --output /tmp/umap-parity-full
	PARITY_CORPUS=/tmp/umap-parity-full go test ./internal/parity
