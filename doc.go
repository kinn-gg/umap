// Package umap provides an allocation-conscious Go implementation of Uniform
// Manifold Approximation and Projection, including dense and CSR inputs,
// deterministic fitting, transforms, supervised fitting, densMAP, and
// checksummed model serialization.
//
// Start with DefaultConfig, change only the parameters needed by the workload,
// and construct an immutable reducer with New. Fitted models are safe for
// concurrent transforms. The compatibility matrix, tuning advice, and
// benchmark methodology are maintained in the repository documentation.
package umap
