package parity

import "testing"

func fixture() *Artifact {
	return &Artifact{SchemaVersion: 1, Name: "test", Suite: "fast", Parameters: Parameters{Neighbors: 2, Components: 2, Metric: "euclidean", MinDist: .1, Spread: 1, LocalConnectivity: 1, Epochs: 20}, Input: Matrix{Rows: 5, Columns: 2, Data: []float64{0, 0, 1, 0, 0, 1, 1, 1, 2, 2}}, KNN: KNN{Indices: []int{0, 1, 1, 0, 2, 0, 3, 1, 4, 3}, Distances: []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1.414}}, SmoothKNN: SmoothKNN{Rho: []float64{1, 1, 1, 1, 1.414}, Sigma: []float64{.1, .1, .1, .1, .1}}, Graph: []Edge{{0, 1, 1}, {0, 2, .5}, {1, 3, .75}}, Initialization: []float64{-1, -1, 1, -1, -1, 1, 1, 1, 2, 2}, Embedding: []float64{-1, -1, 1, -1, -1, 1, 1, 1, 2, 2}}
}

func clone(a *Artifact) *Artifact {
	b := *a
	b.KNN = KNN{append([]int(nil), a.KNN.Indices...), append([]float64(nil), a.KNN.Distances...)}
	b.SmoothKNN = SmoothKNN{append([]float64(nil), a.SmoothKNN.Rho...), append([]float64(nil), a.SmoothKNN.Sigma...)}
	b.Graph = append([]Edge(nil), a.Graph...)
	b.Initialization = append([]float64(nil), a.Initialization...)
	b.Embedding = append([]float64(nil), a.Embedding...)
	return &b
}

func TestCompareEqual(t *testing.T) {
	a := fixture()
	r := Compare(a, clone(a), DefaultThresholds)
	if !r.Passed {
		t.Fatalf("unexpected failure: %+v", r)
	}
}
func TestCompareReportsFirstDivergentStage(t *testing.T) {
	a, b := fixture(), clone(fixture())
	b.SmoothKNN.Rho[3] += .1
	b.Embedding[0] += 100
	r := Compare(a, b, DefaultThresholds)
	if r.Passed || r.Stage != "smooth_knn" {
		t.Fatalf("got %+v", r)
	}
}
func TestCompareRotationInvariant(t *testing.T) {
	a, b := fixture(), clone(fixture())
	for i := 0; i < len(b.Embedding); i += 2 {
		x, y := b.Embedding[i], b.Embedding[i+1]
		b.Embedding[i], b.Embedding[i+1] = -y, x
	}
	r := Compare(a, b, DefaultThresholds)
	if !r.Passed {
		t.Fatalf("rotation should pass: %+v", r)
	}
}

func TestCompareReflectionInvariant(t *testing.T) {
	a, b := fixture(), clone(fixture())
	for i := 1; i < len(b.Embedding); i += 2 {
		b.Embedding[i] = -b.Embedding[i]
	}
	r := Compare(a, b, DefaultThresholds)
	if !r.Passed {
		t.Fatalf("reflection should pass: %+v", r)
	}
}
