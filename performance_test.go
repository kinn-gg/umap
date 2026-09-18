package umap

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"
)

type fitStageFixture struct {
	x          Dense
	neighbors  Neighbors
	rho, sigma []float32
	graph      Graph
	initial    []float32
}

func TestFitABConcurrentCacheIsBitIdentical(t *testing.T) {
	const spread, minDist = 1.23456789, .23456789
	wantA, wantB := fitAB(spread, minDist)
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			a, b := FitAB(spread, minDist)
			if math.Float64bits(a) != math.Float64bits(wantA) || math.Float64bits(b) != math.Float64bits(wantB) {
				t.Errorf("FitAB() = (%v, %v), want bit-identical (%v, %v)", a, b, wantA, wantB)
			}
		}()
	}
	wg.Wait()
}

func newFitStageFixture(tb testing.TB, rows int) fitStageFixture {
	tb.Helper()
	const dimensions, k = 16, 15
	data := make([]float32, rows*dimensions)
	for i := range data {
		data[i] = float32(math.Sin(float64(i)*.17) + math.Cos(float64(i)*.03))
	}
	x, err := NewDense(data, rows, dimensions)
	if err != nil {
		tb.Fatal(err)
	}
	neighbors, err := ExactNeighborsWithOptions(context.Background(), x, k, NewMetric(Euclidean), ExactOptions{Workers: 1})
	if err != nil {
		tb.Fatal(err)
	}
	rho, sigma := SmoothKNN(neighbors, 1, 1)
	graph := FuzzyGraph(neighbors, rho, sigma, 1)
	return fitStageFixture{x, neighbors, rho, sigma, graph, RandomEmbedding(rows, 2, 42)}
}

// BenchmarkFitStages makes the fixed costs of the small-input fit path visible.
// Keep the row counts and fit parameters aligned with benchmarks/README.md and
// cmd/umap-benchmark so profiles and matched Go/Python results are comparable.
func BenchmarkFitStages(b *testing.B) {
	for _, rows := range []int{64, 256} {
		fixture := newFitStageFixture(b, rows)
		b.Run(fmt.Sprintf("rows/%d", rows), func(b *testing.B) {
			b.Run("curve-fit/cold", func(b *testing.B) {
				for range b.N {
					fitAB(1, .1)
				}
			})
			b.Run("curve-fit/cached", func(b *testing.B) {
				FitAB(1, .1)
				b.ResetTimer()
				for range b.N {
					FitAB(1, .1)
				}
			})
			b.Run("neighbors", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					if _, err := ExactNeighborsWithOptions(context.Background(), fixture.x, 15, NewMetric(Euclidean), ExactOptions{Workers: 1}); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("smooth-knn", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					SmoothKNN(fixture.neighbors, 1, 1)
				}
			})
			b.Run("graph", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					FuzzyGraph(fixture.neighbors, fixture.rho, fixture.sigma, 1)
				}
			})
			b.Run("random-init", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					RandomEmbedding(rows, 2, 42)
				}
			})
			b.Run("layout", func(b *testing.B) {
				b.ReportAllocs()
				a, bb := FitAB(1, .1)
				b.ResetTimer()
				for range b.N {
					if _, err := optimizeLayoutContext(context.Background(), fixture.initial, fixture.graph, 2, 100, 1, float32(a), float32(bb), 1, 5, 42); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("clone-input", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					if _, err := cloneMatrix(fixture.x); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}
