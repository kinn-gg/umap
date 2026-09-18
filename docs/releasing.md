# Releases and compatibility

This module follows semantic versioning. Before v1, minor versions may change
public APIs. Starting at v1.0.0, incompatible API or model-format changes require
a new major version; additions use a minor version and compatible fixes a patch.
The decoder may retain support for older model formats beyond that minimum.

## Release checklist

1. Confirm every issue in the target milestone is closed and the compatibility,
   race, fuzz, benchmark, and documentation checks pass on the release commit.
2. Review `LICENSE`, `NOTICE`, dependency licenses, and the pinned Python
   reference. This implementation is clean-room Go code informed by published
   algorithms and reference-compatible behavior; it does not copy Python code.
3. Tag the reviewed merge commit with an annotated semantic version tag such as
   `v1.0.0` and push it. Never tag v1.0.0 before the M5 issues and checks finish.
4. The release workflow reruns tests, verifies the tag, creates reproducible
   source archives, and publishes checksums in the GitHub release.

Go consumers install a release with `go get github.com/kinn-gg/umap@v1.0.0`.
