package umap

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
	"testing"
)

var benchmarkFitA, benchmarkFitB float64

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

func TestFitABRepresentativeGrid(t *testing.T) {
	tests := []struct {
		spread, minDist float64
		a, b            float64
	}{
		{1, .1, 0x1.93b291d9ba1b7p+00, 0x1.ca4567aac83bbp-01},
		{.5, .05, 0x1.1b7669621d1a5p+01, 0x1.0ed87c2ae1406p-01},
		{1, .5, 0x1.05ec66691ff58p+00, 0x1.067fbf204e5cbp+00},
		{2, .1, 0x1.1c2bf1b26bd4ap+00, 0x1.2f4dc9e6de7d6p-01},
	}
	for i, tt := range tests {
		a, b := fitAB(tt.spread, tt.minDist)
		if i == 0 && (math.Float64bits(a) != math.Float64bits(tt.a) || math.Float64bits(b) != math.Float64bits(tt.b)) {
			t.Errorf("default fitAB() = (%v, %v), want bit-identical (%v, %v)", a, b, tt.a, tt.b)
		}
		if math.Abs(a-tt.a) > 1e-12 || math.Abs(b-tt.b) > 1e-12 {
			t.Errorf("fitAB(%v, %v) = (%v, %v), want (%v, %v) within 1e-12", tt.spread, tt.minDist, a, b, tt.a, tt.b)
		}
	}
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

// BenchmarkLayoutOptimization isolates the layout hot path at the matched
// end-to-end sizes, including the high-neighbor and multi-component cases from
// issue #45. Keep these parameters aligned with benchmarks/README.md.
func BenchmarkLayoutOptimization(b *testing.B) {
	a, bb := FitAB(1, .1)
	for _, tc := range []struct {
		rows, neighbors, components int
	}{
		{64, 15, 2},
		{256, 15, 2},
		{256, 50, 2},
		{256, 15, 8},
		{256, 50, 8},
		{1024, 15, 2},
	} {
		fixture := newLayoutBenchmarkFixture(b, tc.rows, tc.neighbors)
		initial := RandomEmbedding(tc.rows, tc.components, 42)
		name := fmt.Sprintf("rows/%d/neighbors/%d/components/%d", tc.rows, tc.neighbors, tc.components)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				if _, err := optimizeLayoutContext(context.Background(), initial, fixture.graph, tc.components, 100, 1, float32(a), float32(bb), 1, 5, 42); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkFuzzyGraph covers the row and neighbor-count matrix from issue #37.
// Fixture construction is excluded so the benchmark isolates graph building.
func BenchmarkFuzzyGraph(b *testing.B) {
	for _, rows := range []int{64, 256, 1024} {
		for _, k := range []int{15, 50} {
			fixture := newLayoutBenchmarkFixture(b, rows, k)
			name := fmt.Sprintf("rows/%d/neighbors/%d", rows, k)
			b.Run(name, func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					FuzzyGraph(fixture.neighbors, fixture.rho, fixture.sigma, 1)
				}
			})
		}
	}
}

// BenchmarkFitStages makes the fixed costs of the small-input fit path visible.
// Keep the row counts and fit parameters aligned with benchmarks/README.md and
// cmd/umap-benchmark so profiles and matched Go/Python results are comparable.
func BenchmarkFitStages(b *testing.B) {
	for _, rows := range []int{64, 256} {
		fixture := newFitStageFixture(b, rows)
		b.Run(fmt.Sprintf("rows/%d", rows), func(b *testing.B) {
			for _, tc := range []struct {
				name            string
				spread, minDist float64
			}{
				{"default", 1, .1},
				{"non-default", 1.23456789, .23456789},
			} {
				b.Run("curve-fit/cold/"+tc.name, func(b *testing.B) {
					for range b.N {
						benchmarkFitA, benchmarkFitB = fitAB(tc.spread, tc.minDist)
					}
				})
			}
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

func sparseTextBenchmarkMatrix(tb testing.TB) CSR {
	tb.Helper()
	const rows, dimensions = 1000, 5000
	const threshold uint64 = 167772
	rng := sparseBenchmarkRNG{42}
	values := make([]float32, 0, 50000)
	columns := make([]uint32, 0, 50000)
	offsets := make([]uint64, rows+1)
	for row := range rows {
		for column := range dimensions {
			value := float32(float64(rng.next()>>40)/float64(uint64(1)<<24)*2 - 1)
			if rng.next()>>40 >= threshold {
				continue
			}
			values = append(values, value)
			columns = append(columns, uint32(column))
		}
		offsets[row+1] = uint64(len(values))
	}
	x, err := NewCSR(values, columns, offsets, rows, dimensions)
	if err != nil {
		tb.Fatal(err)
	}
	return x
}

type sparseBenchmarkRNG struct{ state uint64 }

func (r *sparseBenchmarkRNG) next() uint64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// BenchmarkSparseTextFitStages isolates every stage requested by issue #36
// using the representative 1,000 x 5,000, 1%-dense cosine shape.
func BenchmarkSparseTextFitStages(b *testing.B) {
	x := sparseTextBenchmarkMatrix(b)
	neighbors, err := ExactNeighborsWithOptions(context.Background(), x, 15, NewMetric(Cosine), ExactOptions{Workers: 1})
	if err != nil {
		b.Fatal(err)
	}
	rho, sigma := SmoothKNN(neighbors, 1, 1)
	graph := FuzzyGraph(neighbors, rho, sigma, 1)
	initial := RandomEmbedding(1000, 2, 42)
	a, bb := FitAB(1, .1)
	stages := []struct {
		name string
		fn   func() error
	}{
		{"csr-validation", func() error { _, err := NewCSR(x.values, x.columns, x.offsets, x.rows, x.columnCount); return err }},
		{"clone-input", func() error { _, err := cloneMatrix(x); return err }},
		{"neighbors", func() error {
			_, err := ExactNeighborsWithOptions(context.Background(), x, 15, NewMetric(Cosine), ExactOptions{Workers: 1})
			return err
		}},
		{"smooth-knn", func() error { SmoothKNN(neighbors, 1, 1); return nil }},
		{"graph", func() error { FuzzyGraph(neighbors, rho, sigma, 1); return nil }},
		{"random-init", func() error { RandomEmbedding(1000, 2, 42); return nil }},
		{"layout", func() error {
			_, err := optimizeLayoutContext(context.Background(), initial, graph, 2, 100, 1, float32(a), float32(bb), 1, 5, 42)
			return err
		}},
	}
	for _, stage := range stages {
		b.Run(stage.name, func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				if err := stage.fn(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkSparseTextFitEndToEnd(b *testing.B) {
	x := sparseTextBenchmarkMatrix(b)
	seed := uint64(42)
	cfg := DefaultConfig()
	cfg.Epochs, cfg.Init, cfg.Workers, cfg.Deterministic = 100, RandomInit, 1, true
	cfg.Seed, cfg.Metric = &seed, NewMetric(Cosine)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		u, err := New(cfg)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := u.Fit(context.Background(), x, NoTarget()); err != nil {
			b.Fatal(err)
		}
	}
}

func densityBenchmarkMatrix(tb testing.TB, density float64) CSR {
	tb.Helper()
	const rows, dimensions = 256, 16
	rng := sparseBenchmarkRNG{42}
	values := make([]float32, 0, int(rows*dimensions*density)+rows)
	columns := make([]uint32, 0, cap(values))
	offsets := make([]uint64, rows+1)
	threshold := uint64(density * float64(uint64(1)<<24))
	for row := range rows {
		for column := range dimensions {
			value := float32(float64(rng.next()>>40)/float64(uint64(1)<<24)*2 - 1)
			if rng.next()>>40 >= threshold {
				continue
			}
			values = append(values, value)
			columns = append(columns, uint32(column))
		}
		offsets[row+1] = uint64(len(values))
	}
	x, err := NewCSR(values, columns, offsets, rows, dimensions)
	if err != nil {
		tb.Fatal(err)
	}
	return x
}

func densityBenchmarkChecksum(x CSR) string {
	h := sha256.New()
	var word [8]byte
	for _, value := range x.values {
		binary.LittleEndian.PutUint32(word[:4], math.Float32bits(value))
		_, _ = h.Write(word[:4])
	}
	for _, column := range x.columns {
		binary.LittleEndian.PutUint64(word[:], uint64(column))
		_, _ = h.Write(word[:])
	}
	for _, offset := range x.offsets {
		binary.LittleEndian.PutUint64(word[:], offset)
		_, _ = h.Write(word[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func TestDensityBenchmarkFixturesMatchSuite(t *testing.T) {
	want := map[float64]string{
		.05: "b5c2b256959885a3ea2aae1794d80f95ddd62541672bfc1b6fc34dc4a0d89c06",
		.5:  "cdd74ad6b68281319d82032a7b42ef19fb7b524ff4e2d14b3b95fbf98150b15b",
	}
	for density, checksum := range want {
		if got := densityBenchmarkChecksum(densityBenchmarkMatrix(t, density)); got != checksum {
			t.Errorf("density %g checksum = %s, want matched-suite checksum %s", density, got, checksum)
		}
	}
}

// BenchmarkDensityFitStages uses the exact generator, seed, shape, and fit
// parameters of the matched synthetic/density cases from cmd/umap-benchmark.
func BenchmarkDensityFitStages(b *testing.B) {
	for _, density := range []float64{.05, .5} {
		x := densityBenchmarkMatrix(b, density)
		neighbors, err := ExactNeighborsWithOptions(context.Background(), x, 15, NewMetric(Euclidean), ExactOptions{Workers: 1})
		if err != nil {
			b.Fatal(err)
		}
		rho, sigma := SmoothKNN(neighbors, 1, 1)
		graph := FuzzyGraph(neighbors, rho, sigma, 1)
		initial := RandomEmbedding(256, 2, 42)
		a, bb := FitAB(1, .1)
		b.Run(fmt.Sprintf("density/%.2g/nonzeros/%d/edges/%d", density, len(x.values), len(graph.Edges)), func(b *testing.B) {
			stages := []struct {
				name string
				fn   func() error
			}{
				{"csr-validation", func() error { _, err := NewCSR(x.values, x.columns, x.offsets, x.rows, x.columnCount); return err }},
				{"clone-input", func() error { _, err := cloneMatrix(x); return err }},
				{"neighbors", func() error {
					_, err := ExactNeighborsWithOptions(context.Background(), x, 15, NewMetric(Euclidean), ExactOptions{Workers: 1})
					return err
				}},
				{"smooth-knn", func() error { SmoothKNN(neighbors, 1, 1); return nil }},
				{"graph", func() error { FuzzyGraph(neighbors, rho, sigma, 1); return nil }},
				{"random-init", func() error { RandomEmbedding(256, 2, 42); return nil }},
				{"layout", func() error {
					_, err := optimizeLayoutContext(context.Background(), initial, graph, 2, 100, 1, float32(a), float32(bb), 1, 5, 42)
					return err
				}},
			}
			for _, stage := range stages {
				b.Run(stage.name, func(b *testing.B) {
					b.ReportAllocs()
					for range b.N {
						if err := stage.fn(); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
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
	for _, components := range []int{2, 8} {
		initial := RandomEmbedding(fixture.graph.Vertices, components, 42)
		run := func(ctx context.Context) ([]float32, error) {
			return optimizeLayoutContext(ctx, initial, fixture.graph, components, 20, 1, float32(a), float32(b), 1, 5, 42)
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
				t.Fatalf("%d-component layout differs at coordinate %d: got %g, want %g", components, i, got[i], want[i])
			}
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := run(ctx); err != context.Canceled {
			t.Fatalf("cancellation error = %v, want %v", err, context.Canceled)
		}
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
