package umap

import (
	"context"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

type SearchAlgorithm uint8

const (
	SearchAuto SearchAlgorithm = iota
	SearchExact
	SearchNNDescent
)

type SearchProgress struct {
	Algorithm                SearchAlgorithm
	Iteration, MaxIterations int
	Completed, Total         int64
}

type ProgressFunc func(SearchProgress)

type ExactOptions struct {
	// BlockSize bounds the number of candidate rows visited at a time. If zero,
	// it is derived from MemoryBudget, or defaults to 256.
	BlockSize    int
	MemoryBudget int64
	Workers      int
	Progress     ProgressFunc
}

type NNDescentOptions struct {
	Seed              uint64
	MaxIterations     int
	CandidatePoolSize int
	ConvergenceDelta  float64
	Progress          ProgressFunc
	Workers           int
}

type NeighborSearchOptions struct {
	Algorithm    SearchAlgorithm
	MemoryBudget int64
	Exact        ExactOptions
	Approximate  NNDescentOptions
}

// GraphMemoryEstimate describes the long-lived and peak temporary storage used
// by neighbor search. InputBytes excludes slice/header overhead and is zero when
// the input ownership remains with the caller.
type GraphMemoryEstimate struct {
	InputBytes, NeighborBytes, GraphBytes, TemporaryBytes, PeakBytes int64
}

func EstimateGraphMemory(x Matrix, k, workers int, algorithm SearchAlgorithm, blockSize int) GraphMemoryEstimate {
	n, _ := x.Shape()
	if workers < 1 {
		workers = 1
	}
	if blockSize < 1 {
		blockSize = 256
	}
	blockSize = min(blockSize, n)
	input := int64(0)
	switch v := x.(type) {
	case Dense:
		input = int64(len(v.data)) * 4
	case CSR:
		input = int64(len(v.values))*4 + int64(len(v.columns))*4 + int64(len(v.offsets))*8
	}
	indexBytes := int64(unsafe.Sizeof(int(0)))
	neighbors := int64(n*k) * (indexBytes + 4)
	// FuzzyGraph retains at most two directed Edge values per neighbor while
	// its sorted directed memberships are live.
	graph := int64(2*n*k) * int64(unsafe.Sizeof(Edge{}))
	temporary := int64(workers*blockSize) * 8
	if algorithm == SearchNNDescent {
		temporary = int64(n*k)*indexBytes + int64(n)*4 + int64(workers*k*k)*(indexBytes+4)
	}
	return GraphMemoryEstimate{input, neighbors, graph, temporary, input + neighbors + graph + temporary}
}

func ExactNeighborsWithOptions(ctx context.Context, x Matrix, k int, metric Metric, opts ExactOptions) (Neighbors, error) {
	n, _ := x.Shape()
	if n < 2 || k < 1 || k >= n {
		return Neighbors{}, validationf("neighbors must be in [1, rows)")
	}
	if opts.BlockSize < 0 || opts.MemoryBudget < 0 || opts.Workers < 0 {
		return Neighbors{}, validationf("exact-search limits cannot be negative")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	block := opts.BlockSize
	if block <= 0 && opts.MemoryBudget > 0 {
		budgetBlock := opts.MemoryBudget / 8
		if budgetBlock < 1 {
			budgetBlock = 1
		}
		if budgetBlock > int64(n) {
			block = n
		} else {
			block = int(budgetBlock)
		}
	}
	if block <= 0 {
		block = 256
	}
	if block > n {
		block = n
	}
	out := newNeighborHeap(n, k)
	workers := exactWorkers(x, metric, opts.Workers)
	if csr, ok := x.(CSR); ok && metric.Kind == Cosine && opts.MemoryBudget == 0 && workers == 1 {
		return exactSparseCosineNeighbors(ctx, csr, out, block, opts.Progress)
	}
	distance := newMatrixDistanceEvaluator(metric, x)
	if opts.MemoryBudget > 0 {
		workers = min(workers, max(1, int(opts.MemoryBudget/max(1, int64(block)*8))))
	}
	total := int64(n) * int64(n)
	var done atomic.Int64
	for start := 0; start < n; start += block {
		end := min(n, start+block)
		if workers == 1 {
			// Built-in metrics are symmetric. Visit the upper triangle once and
			// insert each result into both rows. Iterating i before j preserves
			// the same ascending candidate order (and therefore tie behavior) as
			// the full matrix scan.
			for i := 0; i < n; i++ {
				if err := ctx.Err(); err != nil {
					return Neighbors{}, err
				}
				for j := max(start, i); j < end; j++ {
					d := float32(distance(i, j))
					out.offer(i, j, d)
					if i != j {
						out.offer(j, i, d)
					}
				}
			}
			completed := done.Add(int64(n * (end - start)))
			if opts.Progress != nil {
				opts.Progress(SearchProgress{Algorithm: SearchExact, Completed: completed, Total: total})
			}
			continue
		}
		if err := parallelRows(ctx, n, workers, func(i int) {
			for j := start; j < end; j++ {
				out.offer(i, j, float32(distance(i, j)))
			}
		}); err != nil {
			return Neighbors{}, err
		}
		completed := done.Add(int64(n * (end - start)))
		if opts.Progress != nil {
			opts.Progress(SearchProgress{Algorithm: SearchExact, Completed: completed, Total: total})
		}
	}
	return out.neighbors(), nil
}

// exactSparseCosineNeighbors uses a transposed CSR index to accumulate only
// non-zero dot products. It still visits candidates in ascending row order, so
// zero-dot ties and seeded downstream behavior are identical to a full scan.
func exactSparseCosineNeighbors(ctx context.Context, x CSR, out *neighborHeap, block int, progress ProgressFunc) (Neighbors, error) {
	n, dimensions := x.Shape()
	norms := make([]float64, n)
	counts := make([]int, dimensions+1)
	for row := range n {
		columns, values := x.Row(row)
		for i, column := range columns {
			value := float64(values[i])
			norms[row] += value * value
			counts[int(column)+1]++
		}
	}
	for column := range dimensions {
		counts[column+1] += counts[column]
	}
	next := append([]int(nil), counts[:dimensions]...)
	postingRows := make([]uint32, len(x.values))
	postingValues := make([]float32, len(x.values))
	for row := range n {
		columns, values := x.Row(row)
		for i, column := range columns {
			position := next[column]
			postingRows[position] = uint32(row)
			postingValues[position] = values[i]
			next[column]++
		}
	}
	dots := make([]float64, n)
	marks := make([]uint32, n)
	generation := uint32(0)
	total := int64(n) * int64(n)
	for start := 0; start < n; start += block {
		end := min(n, start+block)
		for row := 0; row < n; row++ {
			if err := ctx.Err(); err != nil {
				return Neighbors{}, err
			}
			generation++
			columns, values := x.Row(row)
			for i, column := range columns {
				value := float64(values[i])
				for p := counts[column]; p < counts[int(column)+1]; p++ {
					candidate := int(postingRows[p])
					if candidate < start || candidate >= end {
						continue
					}
					if marks[candidate] != generation {
						marks[candidate] = generation
						dots[candidate] = 0
					}
					dots[candidate] += value * float64(postingValues[p])
				}
			}
			for candidate := start; candidate < end; candidate++ {
				dot := 0.0
				if marks[candidate] == generation {
					dot = dots[candidate]
				}
				distance := 1.0
				if norms[row] == 0 && norms[candidate] == 0 {
					distance = 0
				} else if norms[row] != 0 && norms[candidate] != 0 {
					distance -= dot / math.Sqrt(norms[row]*norms[candidate])
				}
				out.offer(row, candidate, float32(distance))
			}
		}
		completed := int64(n * end)
		if progress != nil {
			progress(SearchProgress{Algorithm: SearchExact, Completed: completed, Total: total})
		}
	}
	return out.neighbors(), nil
}

func NNDescent(ctx context.Context, x Matrix, k int, metric Metric, opts NNDescentOptions) (Neighbors, error) {
	n, _ := x.Shape()
	if n < 2 || k < 1 || k >= n {
		return Neighbors{}, validationf("neighbors must be in [1, rows)")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	iterations := opts.MaxIterations
	if iterations <= 0 {
		iterations = 12
	}
	pool := opts.CandidatePoolSize
	if pool <= 0 {
		pool = max(64, 3*k)
	}
	pool = min(pool, n)
	delta := opts.ConvergenceDelta
	if delta <= 0 {
		delta = .001
	}
	h := newNeighborHeap(n, k)
	distance := newMatrixDistanceEvaluator(metric, x)
	workers := boundedWorkers(opts.Workers, n)
	// Seed with self plus a deterministic random sample. Sampling a moderately
	// sized pool greatly improves high-dimensional starts without an RP tree and
	// remains O(n*k) in retained memory.
	if err := parallelRows(ctx, n, workers, func(i int) {
		rng := NewRNG(DeriveSeed(opts.Seed^uint64(i), "nndescent-row"))
		h.offer(i, i, 0)
		start := int(rng.Uint64() % uint64(n))
		step := int(rng.Uint64()%uint64(max(1, n-1))) + 1
		for gcd(step, n) != 1 {
			step = step%n + 1
		}
		for selected, q := 1, 0; selected < pool; q++ {
			j := (start + q*step) % n
			if j != i {
				selected++
				h.offer(i, j, float32(distance(i, j)))
			}
		}
	}); err != nil {
		return Neighbors{}, err
	}
	for iteration := 0; iteration < iterations; iteration++ {
		if err := ctx.Err(); err != nil {
			return Neighbors{}, err
		}
		before := append([]int(nil), h.indices...)
		changes := 0
		// Neighbor propagation: friends-of-friends form the candidate set. The
		// sorted retained lists make traversal and tie behavior reproducible.
		if err := parallelRows(ctx, n, workers, func(i int) {
			row := before[i*k : (i+1)*k]
			for _, j := range row {
				if j < 0 {
					continue
				}
				for _, candidate := range before[j*k : (j+1)*k] {
					if candidate >= 0 {
						h.offer(i, candidate, float32(distance(i, candidate)))
					}
				}
			}
		}); err != nil {
			return Neighbors{}, err
		}
		for i, v := range h.indices {
			if v != before[i] {
				changes++
			}
		}
		if opts.Progress != nil {
			opts.Progress(SearchProgress{Algorithm: SearchNNDescent, Iteration: iteration + 1, MaxIterations: iterations, Completed: int64(iteration + 1), Total: int64(iterations)})
		}
		if float64(changes) <= delta*float64(n*k) {
			break
		}
	}
	return h.neighbors(), nil
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func boundedWorkers(requested, jobs int) int {
	if requested == 0 {
		requested = runtime.GOMAXPROCS(0)
	}
	if requested < 1 {
		requested = 1
	}
	return min(requested, max(1, jobs))
}

// exactWorkers keeps automatic searches serial until there is enough distance
// work to amortize goroutine startup and the parallel path's full-matrix scan.
// Explicit worker counts remain exact (apart from the row-count bound).
func exactWorkers(x Matrix, metric Metric, requested int) int {
	rows, dimensions := x.Shape()
	if requested != 0 {
		return boundedWorkers(requested, rows)
	}
	workers := boundedWorkers(0, rows)
	if workers < 3 {
		return 1
	}
	// Sparse cosine's inverted-index implementation is both sub-quadratic on
	// typical sparse inputs and currently single-worker. Prefer it over the
	// generic parallel full scan.
	if _, ok := x.(CSR); ok && metric.Kind == Cosine {
		return 1
	}
	const minimumParallelWork = int64(2_000_000)
	work := int64(rows) * int64(rows) * int64(max(1, dimensions))
	if work < minimumParallelWork {
		return 1
	}
	return workers
}

func parallelRows(ctx context.Context, rows, workers int, fn func(int)) error {
	if workers <= 1 {
		for i := 0; i < rows; i++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			fn(i)
		}
		return nil
	}
	var wg sync.WaitGroup
	wg.Add(workers)
	for worker := range workers {
		go func() {
			defer wg.Done()
			start := worker * rows / workers
			end := (worker + 1) * rows / workers
			for i := start; i < end; i++ {
				if ctx.Err() != nil {
					return
				}
				fn(i)
			}
		}()
	}
	wg.Wait()
	return ctx.Err()
}

func FindNeighbors(ctx context.Context, x Matrix, k int, metric Metric, opts NeighborSearchOptions) (Neighbors, error) {
	algorithm := opts.Algorithm
	if algorithm == SearchAuto {
		n, _ := x.Shape()
		if n <= 4096 {
			algorithm = SearchExact
		} else {
			algorithm = SearchNNDescent
		}
	}
	if opts.MemoryBudget < 0 {
		return Neighbors{}, validationf("neighbor-search memory budget cannot be negative")
	}
	if opts.MemoryBudget > 0 {
		block := opts.Exact.BlockSize
		estimate := EstimateGraphMemory(x, k, 1, algorithm, block)
		working := estimate.NeighborBytes + estimate.GraphBytes + estimate.TemporaryBytes
		if working > opts.MemoryBudget {
			return Neighbors{}, validationf("neighbor-search memory budget %d is below estimated working set %d", opts.MemoryBudget, working)
		}
	}
	if algorithm == SearchExact {
		return ExactNeighborsWithOptions(ctx, x, k, metric, opts.Exact)
	}
	if algorithm == SearchNNDescent {
		return NNDescent(ctx, x, k, metric, opts.Approximate)
	}
	return Neighbors{}, validationf("unknown neighbor-search algorithm")
}

type neighborHeap struct {
	rows, k   int
	indices   []int
	distances []float32
}

func newNeighborHeap(rows, k int) *neighborHeap {
	h := &neighborHeap{rows, k, make([]int, rows*k), make([]float32, rows*k)}
	for i := range h.indices {
		h.indices[i] = -1
		h.distances[i] = float32(math.Inf(1))
	}
	return h
}
func (h *neighborHeap) offer(row, index int, distance float32) bool {
	base := row * h.k
	for q := 0; q < h.k; q++ {
		if h.indices[base+q] == index {
			return false
		}
	}
	pos := h.k
	for q := 0; q < h.k; q++ {
		oldD, oldI := h.distances[base+q], h.indices[base+q]
		if distance < oldD || (distance == oldD && (index == row || (oldI != row && index < oldI))) {
			pos = q
			break
		}
	}
	if pos == h.k {
		return false
	}
	copy(h.indices[base+pos+1:base+h.k], h.indices[base+pos:base+h.k-1])
	copy(h.distances[base+pos+1:base+h.k], h.distances[base+pos:base+h.k-1])
	h.indices[base+pos] = index
	h.distances[base+pos] = distance
	return true
}
func (h *neighborHeap) neighbors() Neighbors { return Neighbors{h.rows, h.k, h.indices, h.distances} }
