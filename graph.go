package umap

import (
	"math"
	"sort"
)

type Neighbors struct {
	Rows, K   int
	Indices   []int
	Distances []float32
}

func ExactNeighbors(x Matrix, k int, metric Metric) (Neighbors, error) {
	n, d := x.Shape()
	if n < 2 || k < 1 || k >= n {
		return Neighbors{}, validationf("neighbors must be in [1, rows)")
	}
	out := Neighbors{n, k, make([]int, n*k), make([]float32, n*k)}
	sa, sb := make([]float32, d), make([]float32, d)
	type pair struct {
		i int
		d float64
	}
	pairs := make([]pair, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			pairs[j] = pair{j, MatrixDistance(metric, x, i, x, j, sa, sb)}
		}
		sort.Slice(pairs, func(a, b int) bool {
			return pairs[a].d < pairs[b].d || (pairs[a].d == pairs[b].d && pairs[a].i < pairs[b].i)
		})
		for q := 0; q < k; q++ {
			out.Indices[i*k+q] = pairs[q].i
			out.Distances[i*k+q] = float32(pairs[q].d)
		}
	}
	return out, nil
}

func SmoothKNN(n Neighbors, localConnectivity, bandwidth float32) (rho, sigma []float32) {
	rho = make([]float32, n.Rows)
	sigma = make([]float32, n.Rows)
	target := math.Log2(float64(n.K)) * float64(bandwidth)
	for i := 0; i < n.Rows; i++ {
		ds := n.Distances[i*n.K : (i+1)*n.K]
		nonzero := make([]float32, 0, n.K)
		for _, d := range ds {
			if d > 0 {
				nonzero = append(nonzero, d)
			}
		}
		lc := float64(localConnectivity)
		idx := int(math.Floor(lc))
		if len(nonzero) > 0 {
			if idx > 0 {
				if idx <= len(nonzero) {
					rho[i] = nonzero[idx-1]
				} else {
					rho[i] = nonzero[len(nonzero)-1]
				}
			}
			if float64(idx) < lc && idx < len(nonzero) {
				rho[i] += (float32(lc) - float32(idx)) * nonzero[idx]
			}
		}
		lo, hi, mid := 0., math.Inf(1), 1.
		for it := 0; it < 64; it++ {
			sum := 0.
			// nearest_neighbors includes the sample itself at position zero;
			// umap-learn deliberately excludes it from the entropy sum.
			for _, d := range ds[1:] {
				v := float64(d - rho[i])
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
		mean := 0.
		for _, v := range n.Distances {
			mean += float64(v)
		}
		mean /= float64(len(n.Distances))
		floor := 1e-3 * mean
		if rho[i] > 0 {
			local := 0.
			for _, v := range ds {
				local += float64(v)
			}
			local /= float64(len(ds))
			floor = 1e-3 * local
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
	directed := make(map[[2]int]float32, n.Rows*n.K)
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
			directed[[2]int{i, j}] = w
		}
	}
	keys := make(map[[2]int]struct{}, len(directed))
	for k := range directed {
		a, b := k[0], k[1]
		if a > b {
			a, b = b, a
		}
		keys[[2]int{a, b}] = struct{}{}
	}
	edges := make([]Edge, 0, 2*len(keys))
	for k := range keys {
		a, b := k[0], k[1]
		x, y := directed[[2]int{a, b}], directed[[2]int{b, a}]
		prod := x * y
		w := mix*(x+y-prod) + (1-mix)*prod
		if w > 0 {
			edges = append(edges, Edge{a, b, w}, Edge{b, a, w})
		}
	}
	sort.Slice(edges, func(i, j int) bool {
		return edges[i].Head < edges[j].Head || (edges[i].Head == edges[j].Head && edges[i].Tail < edges[j].Tail)
	})
	return Graph{n.Rows, edges}
}
