package umap

import (
	"cmp"
	"math"
	"slices"
	"sort"
)

type Neighbors struct {
	Rows, K   int
	Indices   []int
	Distances []float32
}

func ExactNeighbors(x Matrix, k int, metric Metric) (Neighbors, error) {
	return ExactNeighborsWithOptions(nil, x, k, metric, ExactOptions{})
}

func SmoothKNN(n Neighbors, localConnectivity, bandwidth float32) (rho, sigma []float32) {
	rho = make([]float32, n.Rows)
	sigma = make([]float32, n.Rows)
	target := math.Log2(float64(n.K)) * float64(bandwidth)
	globalMean := 0.
	for _, v := range n.Distances {
		globalMean += float64(v)
	}
	globalMean /= float64(len(n.Distances))
	globalFloor := 1e-3 * globalMean
	lc := float64(localConnectivity)
	idx := int(math.Floor(lc))
	var adjustedStorage [64]float64
	adjusted := adjustedStorage[:]
	if n.K-1 > len(adjusted) {
		adjusted = make([]float64, n.K-1)
	}
	adjusted = adjusted[:n.K-1]
	for i := 0; i < n.Rows; i++ {
		ds := n.Distances[i*n.K : (i+1)*n.K]
		localSum := 0.
		positive, lower, upper, last := 0, float32(0), float32(0), float32(0)
		for _, d := range ds {
			localSum += float64(d)
			if d <= 0 {
				continue
			}
			if positive == idx-1 {
				lower = d
			}
			if positive == idx {
				upper = d
			}
			last = d
			positive++
		}
		if positive > 0 {
			if idx > 0 {
				if idx <= positive {
					rho[i] = lower
				} else {
					rho[i] = last
				}
			}
			if float64(idx) < lc && idx < positive {
				rho[i] += (float32(lc) - float32(idx)) * upper
			}
		}
		for q, d := range ds[1:] {
			adjusted[q] = float64(d - rho[i])
		}
		lo, hi, mid := 0., math.Inf(1), 1.
		for it := 0; it < 64; it++ {
			sum := 0.
			// nearest_neighbors includes the sample itself at position zero;
			// umap-learn deliberately excludes it from the entropy sum.
			for _, v := range adjusted {
				if v > 0 {
					sum += math.Exp(-v / mid)
				} else {
					sum++
				}
			}
			if math.Abs(sum-target) < 1e-5 {
				break
			}
			if sum > target {
				hi = mid
				mid = (lo + hi) / 2
			} else {
				lo = mid
				if math.IsInf(hi, 1) {
					mid *= 2
				} else {
					mid = (lo + hi) / 2
				}
			}
		}
		floor := globalFloor
		if rho[i] > 0 {
			floor = 1e-3 * (localSum / float64(len(ds)))
		}
		if mid < floor {
			mid = floor
		}
		sigma[i] = float32(mid)
	}
	return
}

type Edge struct {
	Head, Tail int
	Weight     float32
}
type Graph struct {
	Vertices int
	Edges    []Edge
}

func FuzzyGraph(n Neighbors, rho, sigma []float32, mix float32) Graph {
	directed := make([]Edge, 0, n.Rows*n.K)
	for i := 0; i < n.Rows; i++ {
		for q := 0; q < n.K; q++ {
			j := n.Indices[i*n.K+q]
			if j == i {
				continue
			}
			d := n.Distances[i*n.K+q] - rho[i]
			w := float32(1)
			if d > 0 && sigma[i] > 0 {
				w = float32(math.Exp(-float64(d / sigma[i])))
			}
			directed = append(directed, Edge{i, j, w})
		}
	}
	sort.Slice(directed, func(i, j int) bool {
		return directed[i].Head < directed[j].Head || (directed[i].Head == directed[j].Head && directed[i].Tail < directed[j].Tail)
	})
	rowOffsets := make([]int, n.Rows+1)
	for _, e := range directed {
		rowOffsets[e.Head+1]++
	}
	for i := range n.Rows {
		rowOffsets[i+1] += rowOffsets[i]
	}
	weight := func(a, b int) float32 {
		start, end := rowOffsets[a], rowOffsets[a+1]
		i := start + sort.Search(end-start, func(i int) bool { return directed[start+i].Tail >= b })
		if i < end && directed[i].Tail == b {
			return directed[i].Weight
		}
		return 0
	}
	edges := make([]Edge, 0, 2*len(directed))
	previousHead, previousTail, forward := -1, -1, float32(0)
	for _, e := range directed {
		a, b := e.Head, e.Tail
		if a != previousHead || b != previousTail {
			previousHead, previousTail, forward = a, b, e.Weight
		}
		reverse := weight(b, a)
		if a > b {
			if reverse > 0 {
				continue
			}
			a, b = b, a
		}
		prod := forward * reverse
		w := mix*(forward+reverse-prod) + (1-mix)*prod
		if w > 0 {
			edges = append(edges, Edge{a, b, w}, Edge{b, a, w})
		}
	}
	// Bucket by head in linear time, then sort only the small per-row spans by
	// tail. Reuse rowOffsets now that reverse membership lookups are complete.
	clear(rowOffsets)
	for _, e := range edges {
		rowOffsets[e.Head+1]++
	}
	for i := range n.Rows {
		rowOffsets[i+1] += rowOffsets[i]
	}
	next := append([]int(nil), rowOffsets[:n.Rows]...)
	for head := range n.Rows {
		for next[head] < rowOffsets[head+1] {
			i := next[head]
			owner := edges[i].Head
			if owner == head {
				next[head]++
				continue
			}
			edges[i], edges[next[owner]] = edges[next[owner]], edges[i]
			next[owner]++
		}
	}
	for head := range n.Rows {
		row := edges[rowOffsets[head]:rowOffsets[head+1]]
		slices.SortFunc(row, func(a, b Edge) int { return cmp.Compare(a.Tail, b.Tail) })
	}
	return Graph{n.Rows, edges}
}
