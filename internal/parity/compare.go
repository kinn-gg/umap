package parity

import (
	"fmt"
	"math"
	"sort"
)

type Thresholds struct {
	KNNAbs, KNNRel, SmoothAbs, SmoothRel, GraphAbs, GraphRel         float64
	InitNRMSE, EmbeddingNRMSE, NeighborOverlap, TrustworthinessDelta float64
}

var DefaultThresholds = Thresholds{KNNAbs: 1e-6, KNNRel: 1e-5, SmoothAbs: 1e-5, SmoothRel: 1e-4, GraphAbs: 1e-6, GraphRel: 1e-4, InitNRMSE: .02, EmbeddingNRMSE: .10, NeighborOverlap: .90, TrustworthinessDelta: .01}

type Result struct {
	Stage   string             `json:"stage"`
	Passed  bool               `json:"passed"`
	Message string             `json:"message"`
	Metrics map[string]float64 `json:"metrics,omitempty"`
}

func Compare(ref, got *Artifact, t Thresholds) Result {
	if ref.Input.Rows != got.Input.Rows || ref.Input.Columns != got.Input.Columns || ref.Parameters != got.Parameters {
		return fail("input", "shape or effective parameters differ")
	}
	if len(ref.KNN.Indices) != len(got.KNN.Indices) {
		return fail("knn", "artifact lengths differ")
	}
	for i := range ref.KNN.Indices {
		if ref.KNN.Indices[i] != got.KNN.Indices[i] {
			return fail("knn", fmt.Sprintf("neighbor %d: got %d, want %d", i, got.KNN.Indices[i], ref.KNN.Indices[i]))
		}
		if !close(ref.KNN.Distances[i], got.KNN.Distances[i], t.KNNAbs, t.KNNRel) {
			return fail("knn", fmt.Sprintf("distance %d: got %.9g, want %.9g", i, got.KNN.Distances[i], ref.KNN.Distances[i]))
		}
	}
	for _, s := range []struct {
		name string
		a, b []float64
	}{{"rho", ref.SmoothKNN.Rho, got.SmoothKNN.Rho}, {"sigma", ref.SmoothKNN.Sigma, got.SmoothKNN.Sigma}} {
		if i, ok := firstFloatDiff(s.a, s.b, t.SmoothAbs, t.SmoothRel); ok {
			return fail("smooth_knn", fmt.Sprintf("%s %d: got %.9g, want %.9g", s.name, i, s.b[i], s.a[i]))
		}
	}
	if msg := graphDiff(ref.Graph, got.Graph, t); msg != "" {
		return fail("fuzzy_graph", msg)
	}
	initErr := alignedNRMSE(ref.Initialization, got.Initialization, ref.Parameters.Components)
	if initErr > t.InitNRMSE {
		return failMetrics("initialization", fmt.Sprintf("aligned normalized RMSE %.6g exceeds %.6g", initErr, t.InitNRMSE), map[string]float64{"aligned_nrmse": initErr})
	}
	refDense, _ := ref.Input.Dense()
	gotDense, _ := got.Input.Dense()
	refTrust := trustworthiness(refDense, ref.Embedding, ref.Input.Rows, ref.Input.Columns, ref.Parameters.Components, ref.Parameters.Neighbors)
	gotTrust := trustworthiness(gotDense, got.Embedding, got.Input.Rows, got.Input.Columns, got.Parameters.Components, got.Parameters.Neighbors)
	overlap := neighborOverlap(ref.Embedding, got.Embedding, ref.Input.Rows, ref.Parameters.Components, ref.Parameters.Neighbors)
	embedErr := alignedNRMSE(ref.Embedding, got.Embedding, ref.Parameters.Components)
	metrics := map[string]float64{"reference_trustworthiness": refTrust, "candidate_trustworthiness": gotTrust, "neighbor_overlap": overlap, "aligned_nrmse": embedErr, "pairwise_stress": pairwiseStress(ref.Embedding, got.Embedding, ref.Input.Rows, ref.Parameters.Components)}
	if gotTrust < refTrust-t.TrustworthinessDelta {
		return failMetrics("embedding", "trustworthiness regression exceeds threshold", metrics)
	}
	if overlap < t.NeighborOverlap {
		return failMetrics("embedding", "neighbor overlap is below threshold", metrics)
	}
	if embedErr > t.EmbeddingNRMSE {
		return failMetrics("embedding", "aligned normalized RMSE exceeds threshold", metrics)
	}
	return Result{Stage: "complete", Passed: true, Message: "all pipeline stages satisfy parity gates", Metrics: metrics}
}

func fail(stage, msg string) Result { return Result{Stage: stage, Passed: false, Message: msg} }
func failMetrics(stage, msg string, m map[string]float64) Result {
	return Result{Stage: stage, Passed: false, Message: msg, Metrics: m}
}
func close(a, b, atol, rtol float64) bool { return math.Abs(a-b) <= atol+rtol*math.Abs(a) }
func firstFloatDiff(a, b []float64, atol, rtol float64) (int, bool) {
	if len(a) != len(b) {
		return 0, true
	}
	for i := range a {
		if !close(a[i], b[i], atol, rtol) {
			return i, true
		}
	}
	return 0, false
}
func graphDiff(a, b []Edge, t Thresholds) string {
	cp := func(x []Edge) []Edge {
		y := append([]Edge(nil), x...)
		sort.Slice(y, func(i, j int) bool { return y[i].Head < y[j].Head || y[i].Head == y[j].Head && y[i].Tail < y[j].Tail })
		return y
	}
	a, b = cp(a), cp(b)
	if len(a) != len(b) {
		return fmt.Sprintf("edge count: got %d, want %d", len(b), len(a))
	}
	for i := range a {
		if a[i].Head != b[i].Head || a[i].Tail != b[i].Tail {
			return fmt.Sprintf("edge %d endpoints differ", i)
		}
		if !close(a[i].Weight, b[i].Weight, t.GraphAbs, t.GraphRel) {
			return fmt.Sprintf("edge (%d,%d) weight: got %.9g, want %.9g", a[i].Head, a[i].Tail, b[i].Weight, a[i].Weight)
		}
	}
	return ""
}

func distances(x []float64, n, d int) [][]float64 {
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, n)
		for j := 0; j < i; j++ {
			s := 0.
			for c := 0; c < d; c++ {
				v := x[i*d+c] - x[j*d+c]
				s += v * v
			}
			v := math.Sqrt(s)
			out[i][j] = v
			out[j][i] = v
		}
	}
	return out
}
func ranks(ds []float64, self int) []int {
	idx := make([]int, 0, len(ds)-1)
	for i := range ds {
		if i != self {
			idx = append(idx, i)
		}
	}
	sort.Slice(idx, func(i, j int) bool { return ds[idx[i]] < ds[idx[j]] || ds[idx[i]] == ds[idx[j]] && idx[i] < idx[j] })
	return idx
}
func trustworthiness(x, z []float64, n, xd, zd, k int) float64 {
	if k >= n/2 {
		k = (n - 1) / 2
	}
	dx, dz := distances(x, n, xd), distances(z, n, zd)
	pen := 0.
	for i := 0; i < n; i++ {
		rx, rz := ranks(dx[i], i), ranks(dz[i], i)
		pos := make([]int, n)
		for p, j := range rx {
			pos[j] = p + 1
		}
		in := make([]bool, n)
		for _, j := range rx[:k] {
			in[j] = true
		}
		for _, j := range rz[:k] {
			if !in[j] {
				pen += float64(pos[j] - k)
			}
		}
	}
	den := float64(n * k * (2*n - 3*k - 1))
	if den <= 0 {
		return 1
	}
	return 1 - 2*pen/den
}
func neighborOverlap(a, b []float64, n, d, k int) float64 {
	if k >= n {
		k = n - 1
	}
	da, db := distances(a, n, d), distances(b, n, d)
	hits := 0
	for i := 0; i < n; i++ {
		ra, rb := ranks(da[i], i), ranks(db[i], i)
		set := map[int]bool{}
		for _, j := range ra[:k] {
			set[j] = true
		}
		for _, j := range rb[:k] {
			if set[j] {
				hits++
			}
		}
	}
	return float64(hits) / float64(n*k)
}
func pairwiseStress(a, b []float64, n, d int) float64 {
	da, db := distances(a, n, d), distances(b, n, d)
	num, den := 0., 0.
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			v := da[i][j] - db[i][j]
			num += v * v
			den += da[i][j] * da[i][j]
		}
	}
	if den == 0 {
		return 0
	}
	return math.Sqrt(num / den)
}

// alignedNRMSE uses the closed-form optimal rotation/reflection after centering.
// UMAP's v1 output is two-dimensional; other dimensions fall back to a
// centered normalized RMSE until a general SVD is needed.
func alignedNRMSE(a, b []float64, d int) float64 {
	n := len(a) / d
	if len(a) != len(b) || n == 0 {
		return math.Inf(1)
	}
	ac, bc := append([]float64(nil), a...), append([]float64(nil), b...)
	for c := 0; c < d; c++ {
		ma, mb := 0., 0.
		for i := 0; i < n; i++ {
			ma += ac[i*d+c]
			mb += bc[i*d+c]
		}
		ma /= float64(n)
		mb /= float64(n)
		for i := 0; i < n; i++ {
			ac[i*d+c] -= ma
			bc[i*d+c] -= mb
		}
	}
	if d == 2 { // Try both determinant signs: orthogonal Procrustes permits reflection.
		align := func(src []float64, reflect bool) []float64 {
			out := append([]float64(nil), src...)
			sxx, sxy := 0., 0.
			for i := 0; i < n; i++ {
				ax, ay := ac[2*i], ac[2*i+1]
				bx, by := out[2*i], out[2*i+1]
				if reflect {
					by = -by
					out[2*i+1] = by
				}
				sxx += ax*bx + ay*by
				sxy += ax*by - ay*bx
			}
			norm := math.Hypot(sxx, sxy)
			if norm > 0 {
				co, si := sxx/norm, sxy/norm
				for i := 0; i < n; i++ {
					x, y := out[2*i], out[2*i+1]
					out[2*i] = co*x + si*y
					out[2*i+1] = -si*x + co*y
				}
			}
			return out
		}
		x, y := align(bc, false), align(bc, true)
		err := func(v []float64) float64 {
			s := 0.
			for i := range ac {
				q := ac[i] - v[i]
				s += q * q
			}
			return s
		}
		if err(y) < err(x) {
			bc = y
		} else {
			bc = x
		}
	}
	num, den := 0., 0.
	for i := range ac {
		v := ac[i] - bc[i]
		num += v * v
		den += ac[i] * ac[i]
	}
	if den == 0 {
		if num == 0 {
			return 0
		}
		return math.Inf(1)
	}
	return math.Sqrt(num / den)
}
