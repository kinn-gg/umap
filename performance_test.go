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
	return newLayoutBenchmarkFixture(tb, rows, 15)
}

func newLayoutBenchmarkFixture(tb testing.TB, rows, k int) fitStageFixture {
	tb.Helper()
	const dimensions = 16
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

// BenchmarkLayoutOptimization isolates the issue #35 hot path at the matched
// end-to-end sizes. Keep these parameters aligned with benchmarks/README.md.
func BenchmarkLayoutOptimization(b *testing.B) {
	a, bb := FitAB(1, .1)
	for _, tc := range []struct {
		rows, neighbors int
	}{
		{64, 15},
		{256, 15},
		{256, 50},
		{1024, 15},
	} {
		fixture := newLayoutBenchmarkFixture(b, tc.rows, tc.neighbors)
		name := fmt.Sprintf("rows/%d/neighbors/%d", tc.rows, tc.neighbors)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				if _, err := optimizeLayoutContext(context.Background(), fixture.initial, fixture.graph, 2, 100, 1, float32(a), float32(bb), 1, 5, 42); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
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

func smoothKNNBenchmarkNeighbors(rows int, duplicated bool) Neighbors {
	const k = 15
	distances := make([]float32, rows*k)
	for i := 0; i < rows; i++ {
		for q := 1; q < k; q++ {
			d := q
			if duplicated {
				d = (q + 1) / 2
			}
			distances[i*k+q] = float32(d) + float32(i%7)*.01
		}
	}
	return Neighbors{Rows: rows, K: k, Distances: distances}
}

// BenchmarkSmoothKNNCalibration isolates the calibration hot path at the
// issue #33 sizes and includes tied nonzero distances.
func BenchmarkSmoothKNNCalibration(b *testing.B) {
	for _, rows := range []int{64, 256, 1024} {
		for _, duplicated := range []bool{false, true} {
			name := "distinct"
			if duplicated {
				name = "duplicated"
			}
			n := smoothKNNBenchmarkNeighbors(rows, duplicated)
			b.Run(fmt.Sprintf("rows/%d/%s", rows, name), func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					SmoothKNN(n, 1, 1)
				}
			})
		}
	}
}

func TestSmoothKNNAllocationGate(t *testing.T) {
	n := smoothKNNBenchmarkNeighbors(256, true)
	if got := testing.AllocsPerRun(100, func() { SmoothKNN(n, 1, 1) }); got > 2 {
		t.Fatalf("SmoothKNN allocations = %g, want at most the two result slices", got)
	}
}

func BenchmarkDenseDistanceDimensions(b *testing.B) {
	for _, dimensions := range []int{2, 16, 128, 512} {
		a := make([]float32, dimensions)
		bb := make([]float32, dimensions)
		for i := range a {
			a[i] = float32(math.Sin(float64(i) * .17))
			bb[i] = float32(math.Cos(float64(i) * .11))
		}
		for _, kind := range []MetricKind{Euclidean, Cosine} {
			metric := NewMetric(kind)
			name := "euclidean"
			if kind == Cosine {
				name = "cosine"
			}
			b.Run(fmt.Sprintf("dimensions/%d/metric/%s", dimensions, name), func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					_ = metric.Distance(a, bb)
				}
			})
		}
	}
}

func BenchmarkDenseFitDimensions(b *testing.B) {
	for _, dimensions := range []int{2, 16, 128, 512} {
		data := make([]float32, 256*dimensions)
		for i := range data {
			data[i] = float32(math.Sin(float64(i)*.17) + math.Cos(float64(i)*.03))
		}
		x, err := NewDense(data, 256, dimensions)
		if err != nil {
			b.Fatal(err)
		}
		cfg := DefaultConfig()
		cfg.Epochs, cfg.Init, cfg.Workers, cfg.Deterministic = 100, RandomInit, 1, true
		seed := uint64(42)
		cfg.Seed = &seed
		b.Run(fmt.Sprintf("dimensions/%d", dimensions), func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				u, err := New(cfg)
				if err != nil {
					b.Fatal(err)
				}
				if _, err := u.Fit(context.Background(), x, NoTarget()); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestLayoutOptimizationDeterministicAndCancellable(t *testing.T) {
	fixture := newLayoutBenchmarkFixture(t, 64, 15)
	a, b := FitAB(1, .1)
	run := func(ctx context.Context) ([]float32, error) {
		return optimizeLayoutContext(ctx, fixture.initial, fixture.graph, 2, 20, 1, float32(a), float32(b), 1, 5, 42)
	}
	want, err := run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	got, err := run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("layout differs at component %d: got %g, want %g", i, got[i], want[i])
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := run(ctx); err != context.Canceled {
		t.Fatalf("cancellation error = %v, want %v", err, context.Canceled)
	}
}

func TestLayoutOptimizationAllocationGate(t *testing.T) {
	fixture := newLayoutBenchmarkFixture(t, 64, 15)
	a, b := FitAB(1, .1)
	allocs := testing.AllocsPerRun(10, func() {
		if _, err := optimizeLayoutContext(context.Background(), fixture.initial, fixture.graph, 2, 10, 1, float32(a), float32(b), 1, 5, 42); err != nil {
			t.Fatal(err)
		}
	})
	if allocs > 5 {
		t.Fatalf("layout allocations = %g, want at most 5", allocs)
	}
}
