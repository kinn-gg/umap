package umap

import (
	"context"
	"math"
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
	Progress     ProgressFunc
}

type NNDescentOptions struct {
	Seed              uint64
	MaxIterations     int
	CandidatePoolSize int
	ConvergenceDelta  float64
	Progress          ProgressFunc
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
	if opts.BlockSize < 0 || opts.MemoryBudget < 0 {
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
	total := int64(n) * int64(n)
	done := int64(0)
	for start := 0; start < n; start += block {
		end := min(n, start+block)
		for i := 0; i < n; i++ {
			for j := start; j < end; j++ {
				out.offer(i, j, float32(MatrixDistance(metric, x, i, x, j, nil, nil)))
			}
		}
		done += int64(n * (end - start))
		if err := ctx.Err(); err != nil {
			return Neighbors{}, err
		}
		if opts.Progress != nil {
			opts.Progress(SearchProgress{Algorithm: SearchExact, Completed: done, Total: total})
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
	rng := NewRNG(opts.Seed)
	seen := make([]uint32, n)
	// Seed with self plus a deterministic random sample. Sampling a moderately
	// sized pool greatly improves high-dimensional starts without an RP tree and
	// remains O(n*k) in retained memory.
	for i := 0; i < n; i++ {
		if err := ctx.Err(); err != nil {
			return Neighbors{}, err
		}
		h.offer(i, i, 0)
		stamp := uint32(i + 1)
		seen[i] = stamp
		selected := 1
		for selected < pool {
			j := int(rng.Uint64() % uint64(n))
			if seen[j] == stamp {
				continue
			}
			seen[j] = stamp
			selected++
			h.offer(i, j, float32(MatrixDistance(metric, x, i, x, j, nil, nil)))
		}
	}
	for iteration := 0; iteration < iterations; iteration++ {
		if err := ctx.Err(); err != nil {
			return Neighbors{}, err
		}
		before := append([]int(nil), h.indices...)
		changes := 0
		// Neighbor propagation: friends-of-friends form the candidate set. The
		// sorted retained lists make traversal and tie behavior reproducible.
		for i := 0; i < n; i++ {
			row := before[i*k : (i+1)*k]
			for _, j := range row {
				if j < 0 {
					continue
				}
				for _, candidate := range before[j*k : (j+1)*k] {
					if candidate >= 0 {
						h.offer(i, candidate, float32(MatrixDistance(metric, x, i, x, candidate, nil, nil)))
					}
				}
			}
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
