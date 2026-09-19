package umap

import (
	"math"
	"reflect"
	"sort"
	"testing"
)

func TestFuzzyGraphPreservesDuplicateMembershipSemantics(t *testing.T) {
	n := Neighbors{
		Rows: 4,
		K:    4,
		Indices: []int{
			0, 2, 1, 2,
			1, 0, 3, 0,
			2, 3, 0, 3,
			3, 1, 2, 1,
		},
		Distances: []float32{
			0, .2, .4, .7,
			0, .3, .5, .8,
			0, .1, .6, .9,
			0, .25, .55, .85,
		},
	}
	rho := []float32{.05, .1, 0, .2}
	sigma := []float32{.7, 1.1, .8, .9}

	want := fuzzyGraphReference(n, rho, sigma, .35)
	got := FuzzyGraph(n, rho, sigma, .35)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("FuzzyGraph() = %#v, want bit-identical %#v", got, want)
	}
}

// fuzzyGraphReference is the pre-optimization implementation retained here to
// lock down ordering, weights, and the first-membership behavior for duplicates.
func fuzzyGraphReference(n Neighbors, rho, sigma []float32, mix float32) Graph {
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
	weight := func(a, b int) float32 {
		i := sort.Search(len(directed), func(i int) bool {
			return directed[i].Head > a || (directed[i].Head == a && directed[i].Tail >= b)
		})
		if i < len(directed) && directed[i].Head == a && directed[i].Tail == b {
			return directed[i].Weight
		}
		return 0
	}
	edges := make([]Edge, 0, 2*len(directed))
	for _, e := range directed {
		a, b := e.Head, e.Tail
		if a > b {
			if weight(b, a) > 0 {
				continue
			}
			a, b = b, a
		}
		x, y := weight(a, b), weight(b, a)
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
