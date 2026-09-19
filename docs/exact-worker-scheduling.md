# Exact-search worker scheduling

Exact neighbor search uses deterministic static row ranges for parallel work.
This avoids the atomic row claim previously paid for every query row and keeps
each worker's writes confined to its assigned output rows.

An explicit `ExactOptions.Workers` value is always honored, bounded only by the
number of rows and the configured memory budget. Automatic selection uses one
worker for workloads below two million row-pair-dimension operations. It also
uses one worker for sparse cosine input so that the inverted-index exact search
is used instead of a quadratic generic scan. At least three available workers
are required before automatic selection chooses the parallel full-matrix path;
with fewer workers, its duplicated symmetric distance evaluations do not have a
reliable advantage over the single-worker triangular scan.

`BenchmarkExactNeighborWorkers` covers dense Euclidean and cosine inputs at
multiple row and dimensionality sizes, plus sparse cosine inputs. It includes
one, two, four, eight, and automatically selected worker counts so changes to
the threshold and scheduler can be evaluated together.

On an Intel i5-12600K with Go 1.27.1, three 500 ms samples of the representative
512-row, 32-dimensional dense Euclidean case measured 4.35-4.39 ms with four
workers before static scheduling and 2.99-3.14 ms afterward (a 28-31% reduction).
Eight-worker time remained within the original 2.70-2.73 ms range, while the
automatic 16-worker result measured 1.75-1.89 ms versus 7.92-7.94 ms with one
worker. The 128-row, 16-dimensional automatic case stays on the serial path and
measures within 3% of an explicit single worker.
