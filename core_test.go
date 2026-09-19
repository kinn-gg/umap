package umap

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestDenseAndCSRViews(t *testing.T) {
	d, err := NewDenseStride([]float32{1, 2, 99, 3, 4}, 2, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if got := d.Row(1)[0]; got != 3 {
		t.Fatalf("row view = %v", got)
	}
	c, err := NewCSR([]float32{2, 3}, []uint32{1, 0}, []uint64{0, 1, 2}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if c.At(0, 1) != 2 || c.At(1, 1) != 0 {
		t.Fatal("bad CSR lookup")
	}
	if _, err := NewCSR([]float32{1, 2}, []uint32{1, 1}, []uint64{0, 2}, 1, 2); err == nil {
		t.Fatal("accepted duplicate CSR column")
	}
}

func TestMetrics(t *testing.T) {
	a, b := []float32{0, 1, 1}, []float32{1, 1, 0}
	tests := []struct {
		m    Metric
		want float64
	}{
		{NewMetric(Euclidean), math.Sqrt2}, {NewMetric(SquaredEuclidean), 2},
		{NewMetric(Manhattan), 2}, {NewMetric(Chebyshev), 1},
		{NewMetric(Hamming), 2.0 / 3}, {NewMetric(Jaccard), 2.0 / 3}, {NewMetric(Dice), .5},
	}
	for _, tc := range tests {
		if got := tc.m.Distance(a, b); math.Abs(got-tc.want) > 1e-7 {
			t.Errorf("metric %v = %g, want %g", tc.m.Kind, got, tc.want)
		}
	}
}

func TestRNGGoldenAndShuffle(t *testing.T) {
	r := NewRNG(42)
	want := []uint64{13679457532755275413, 2949826092126892291, 5139283748462763858}
	for i, w := range want {
		if got := r.Uint64(); got != w {
			t.Fatalf("value %d = %d, want %d", i, got, w)
		}
	}
	a := []int{0, 1, 2, 3, 4}
	NewRNG(7).Shuffle(len(a), func(i, j int) { a[i], a[j] = a[j], a[i] })
	got := ""
	for _, v := range a {
		got += string(rune('0' + v))
	}
	if got != "41302" {
		t.Fatalf("shuffle = %s", got)
	}
}

func TestReferenceFixtureKNNAndSmooth(t *testing.T) {
	b, err := os.ReadFile("parity/fixtures/dense.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Input struct {
			Rows, Columns int
			Data          []float32
		}
		Parameters struct {
			Neighbors int     `json:"n_neighbors"`
			Local     float32 `json:"local_connectivity"`
		}
		KNN struct {
			Indices   []int
			Distances []float32
		}
		Smooth struct{ Rho, Sigma []float32 } `json:"smooth_knn"`
		Graph  []Edge                         `json:"fuzzy_graph"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	x, err := NewDense(f.Input.Data, f.Input.Rows, f.Input.Columns)
	if err != nil {
		t.Fatal(err)
	}
	n, err := ExactNeighbors(x, f.Parameters.Neighbors, NewMetric(Euclidean))
	if err != nil {
		t.Fatal(err)
	}
	for i := range n.Indices {
		if n.Indices[i] != f.KNN.Indices[i] || math.Abs(float64(n.Distances[i]-f.KNN.Distances[i])) > 1e-5 {
			t.Fatalf("kNN mismatch at %d", i)
		}
	}
	rho, sigma := SmoothKNN(n, 1, 1)
	for i := range rho {
		if math.Abs(float64(rho[i]-f.Smooth.Rho[i])) > 1e-5 || math.Abs(float64(sigma[i]-f.Smooth.Sigma[i])) > 2e-4 {
			t.Fatalf("smooth mismatch at %d: got %g/%g want %g/%g", i, rho[i], sigma[i], f.Smooth.Rho[i], f.Smooth.Sigma[i])
		}
	}
	g := FuzzyGraph(n, rho, sigma, 1)
	if len(g.Edges) != len(f.Graph) {
		t.Fatalf("graph has %d edges, want %d", len(g.Edges), len(f.Graph))
	}
	for i := range g.Edges {
		if g.Edges[i].Head != f.Graph[i].Head || g.Edges[i].Tail != f.Graph[i].Tail || math.Abs(float64(g.Edges[i].Weight-f.Graph[i].Weight)) > 2e-5 {
			t.Fatalf("graph mismatch at %d: got %+v want %+v", i, g.Edges[i], f.Graph[i])
		}
	}
}

func TestSmoothKNNDegenerateAndDuplicatedFixtures(t *testing.T) {
	for _, name := range []string{"degenerate", "duplicated"} {
		t.Run(name, func(t *testing.T) {
			b, err := os.ReadFile("parity/fixtures/" + name + ".json")
			if err != nil {
				t.Fatal(err)
			}
			var f struct {
				Input      struct{ Rows int }
				Parameters struct {
					Neighbors int     `json:"n_neighbors"`
					Local     float32 `json:"local_connectivity"`
				}
				KNN struct {
					Indices   []int
					Distances []float32
				}
				Smooth struct{ Rho, Sigma []float32 } `json:"smooth_knn"`
			}
			if err := json.Unmarshal(b, &f); err != nil {
				t.Fatal(err)
			}
			n := Neighbors{Rows: f.Input.Rows, K: f.Parameters.Neighbors, Indices: f.KNN.Indices, Distances: f.KNN.Distances}
			rho, sigma := SmoothKNN(n, f.Parameters.Local, 1)
			for i := range rho {
				if math.Abs(float64(rho[i]-f.Smooth.Rho[i])) > 1e-5 || math.Abs(float64(sigma[i]-f.Smooth.Sigma[i])) > 2e-4 {
					t.Fatalf("smooth mismatch at %d: got %g/%g want %g/%g", i, rho[i], sigma[i], f.Smooth.Rho[i], f.Smooth.Sigma[i])
				}
			}
		})
	}
}

func TestFitDeterministic(t *testing.T) {
	x, _ := NewDense([]float32{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
	seed := uint64(9)
	c := DefaultConfig()
	c.Neighbors = 3
	c.Epochs = 4
	c.Seed = &seed
	c.Deterministic = true
	u, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	_, a, err := u.FitTransform(context.Background(), x, NoTarget())
	if err != nil {
		t.Fatal(err)
	}
	_, b, err := u.FitTransform(context.Background(), x, NoTarget())
	if err != nil {
		t.Fatal(err)
	}
	for i := range a.data {
		if a.data[i] != b.data[i] {
			t.Fatalf("not deterministic at %d", i)
		}
	}
}

func BenchmarkEuclidean(b *testing.B) {
	x := make([]float32, 128)
	y := make([]float32, 128)
	m := NewMetric(Euclidean)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = m.Distance(x, y)
	}
}

func TestFastPowfAccuracy(t *testing.T) {
	for _, p := range []float32{0.5, 0.8950609, 1, 1.5} {
		evaluator := newFastPowfEvaluator(p)
		for exponent := -20; exponent <= 20; exponent++ {
			for mantissa := float32(1); mantissa < 2; mantissa += 1.0 / 32 {
				x := float32(math.Ldexp(float64(mantissa), exponent))
				want := float32(math.Pow(float64(x), float64(p)))
				got := evaluator.eval(x)
				if relative := math.Abs(float64(got-want)) / float64(want); relative > 2e-5 {
					t.Fatalf("fastPowfEvaluator(%g, %g) = %g, want %g (relative error %g)", x, p, got, want, relative)
				}
			}
		}
	}
}

func TestLayoutClampSpecialization(t *testing.T) {
	tests := []struct {
		name string
		in   float32
		want float32
	}{
		{"negative overflow", -5, -4},
		{"negative boundary", -4, -4},
		{"negative zero", float32(math.Copysign(0, -1)), float32(math.Copysign(0, -1))},
		{"positive zero", 0, 0},
		{"positive boundary", 4, 4},
		{"positive overflow", 5, 4},
		{"negative infinity", float32(math.Inf(-1)), -4},
		{"positive infinity", float32(math.Inf(1)), 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := clamp(tt.in, -4, 4)
			if math.Float32bits(got) != math.Float32bits(tt.want) {
				t.Fatalf("clamp(%v, -4, 4) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
	nan := float32(math.NaN())
	if got := clamp(nan, -4, 4); !math.IsNaN(float64(got)) {
		t.Fatalf("clamp(NaN, -4, 4) = %v, want NaN", got)
	}
}

func BenchmarkExactNeighbors(b *testing.B) {
	data := make([]float32, 256*16)
	x, _ := NewDense(data, 256, 16)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ExactNeighbors(x, 15, NewMetric(Euclidean))
	}
}
