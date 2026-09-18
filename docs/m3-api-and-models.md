# Milestone 3 API, targets, transforms, and model files

`Fit` and `FitTransform` validate the input and target, build the fuzzy graph,
apply optional target information, and retain an owned copy of the training
matrix. Caller buffers can therefore be reused after fitting. `Model.Graph`
returns a copy and embeddings expose read-only accessors.

Categorical targets weaken edges between known, different classes. The unknown
value supplied to `NewCategories` leaves incident edges unsupervised. Continuous
targets support L1 and squared-L2 similarity. A target weight of zero follows
the exact unsupervised path.

`Transform` searches the retained training observations, initializes each new
coordinate from distance-weighted fitted neighbors, then performs
transform-specific attraction epochs without changing the fitted embedding.
Queries are independent, so splitting the same observations into batches does
not change seeded results. `TransformInto` accepts caller-owned output storage.

`Model.MarshalBinary` emits a versioned, checksummed format. `UnmarshalModel`
rejects unknown versions, bad dimensions, malformed sparse matrices, invalid
graphs, non-finite values, checksum failures, and payload declarations over
256 MiB before allocating for the declared payload.
