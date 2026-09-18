package umap

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"
)

func TestExactNeighborsBlockedTiesAndSelf(t *testing.T) {
	x, _ := NewDense([]float32{0, 0, 1, -1}, 4, 1)
	var progress []int64
	n, err := ExactNeighborsWithOptions(context.Background(), x, 3, NewMetric(Euclidean), ExactOptions{BlockSize: 1, Progress: func(p SearchProgress) { progress = append(progress, p.Completed) }})
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 1, 2, 1, 0, 2, 2, 0, 1, 3, 0, 1}
	for i := range want {
		if n.Indices[i] != want[i] {
			t.Fatalf("index %d = %d, want %d", i, n.Indices[i], want[i])
		}
	}
	if len(progress) != 4 || progress[len(progress)-1] != 16 {
		t.Fatalf("progress = %v", progress)
	}
}

func TestSparseDistancesAndNeighborsMatchDense(t *testing.T) {
	denseData := []float32{0, 2, 0, 4, 1, 0, 3, 0, 0, 2, 3, 0, 1, 0, 0, 4}
	dense, _ := NewDense(denseData, 4, 4)
	csr, err := NewCSR(
		[]float32{2, 4, 1, 3, 2, 3, 1, 4},
		[]uint32{1, 3, 0, 2, 1, 2, 0, 3},
		[]uint64{0, 2, 4, 6, 8}, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	metrics := []MetricKind{Euclidean, SquaredEuclidean, Manhattan, Chebyshev, Cosine, Correlation, Canberra, BrayCurtis, Hamming, Jaccard, Dice}
	for _, kind := range metrics {
		metric := NewMetric(kind)
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				want := MatrixDistance(metric, dense, i, dense, j, nil, nil)
				for name, got := range map[string]float64{"sparse-sparse": MatrixDistance(metric, csr, i, csr, j, nil, nil), "dense-sparse": MatrixDistance(metric, dense, i, csr, j, nil, nil), "sparse-dense": MatrixDistance(metric, csr, i, dense, j, nil, nil)} {
					if math.Abs(got-want) > 1e-12 {
						t.Fatalf("%v %s (%d,%d) = %g, want %g", kind, name, i, j, got, want)
					}
				}
			}
		}
		a, _ := ExactNeighbors(dense, 3, metric)
		b, _ := ExactNeighbors(csr, 3, metric)
		for i := range a.Indices {
			if a.Indices[i] != b.Indices[i] || math.Abs(float64(a.Distances[i]-b.Distances[i])) > 1e-6 {
				t.Fatalf("%v neighbor %d differs", kind, i)
			}
		}
	}
}

func TestNNDescentReproducibleRecallProgressAndCancellation(t *testing.T) {
	const rows, dimensions, k = 240, 8, 10
	data := make([]float32, rows*dimensions)
	rng := NewRNG(123)
	for i := range data {
		data[i] = rng.Float32()
	}
	x, _ := NewDense(data, rows, dimensions)
	exact, _ := ExactNeighbors(x, k, NewMetric(Euclidean))
	progress := 0
	opts := NNDescentOptions{Seed: 42, MaxIterations: 15, CandidatePoolSize: 48, ConvergenceDelta: .0001, Progress: func(SearchProgress) { progress++ }}
	a, err := NNDescent(context.Background(), x, k, NewMetric(Euclidean), opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := NNDescent(context.Background(), x, k, NewMetric(Euclidean), opts)
	if err != nil {
		t.Fatal(err)
	}
	if progress == 0 {
		t.Fatal("no progress callbacks")
	}
	hits := 0
	for i := 0; i < rows; i++ {
		set := map[int]bool{}
		for _, v := range exact.Indices[i*k : (i+1)*k] {
			set[v] = true
		}
		for q, v := range a.Indices[i*k : (i+1)*k] {
			if v != b.Indices[i*k+q] || a.Distances[i*k+q] != b.Distances[i*k+q] {
				t.Fatal("fixed seed is not reproducible")
			}
			if set[v] {
				hits++
			}
		}
	}
	recall := float64(hits) / float64(rows*k)
	if recall < .90 {
		t.Fatalf("recall@%d = %.3f, want >= .90", k, recall)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NNDescent(ctx, x, k, NewMetric(Euclidean), opts); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel error = %v", err)
	}
}

func TestGraphMemoryEstimateUsesSparseStorage(t *testing.T) {
	dense, _ := NewDense(make([]float32, 100*1000), 100, 1000)
	sparse, _ := NewCSR(make([]float32, 100), make([]uint32, 100), func() []uint64 {
		v := make([]uint64, 101)
		for i := range v {
			v[i] = uint64(i)
		}
		return v
	}(), 100, 1000)
	d := EstimateGraphMemory(dense, 15, 1, SearchExact, 32)
	s := EstimateGraphMemory(sparse, 15, 1, SearchExact, 32)
	if s.InputBytes >= d.InputBytes/100 {
		t.Fatalf("sparse input estimate %d is not proportional to nnz; dense=%d", s.InputBytes, d.InputBytes)
	}
	if s.NeighborBytes != d.NeighborBytes || s.PeakBytes != s.InputBytes+s.NeighborBytes+s.GraphBytes+s.TemporaryBytes {
		t.Fatal("invalid memory accounting")
	}
}

func TestNeighborSearchMemoryBudget(t *testing.T) {
	x, _ := NewDense(make([]float32, 32*4), 32, 4)
	estimate := EstimateGraphMemory(x, 5, 1, SearchExact, 8)
	working := estimate.NeighborBytes + estimate.GraphBytes + estimate.TemporaryBytes
	if _, err := FindNeighbors(context.Background(), x, 5, NewMetric(Euclidean), NeighborSearchOptions{Algorithm: SearchExact, MemoryBudget: working - 1, Exact: ExactOptions{BlockSize: 8}}); err == nil {
		t.Fatal("accepted a memory budget below the predicted working set")
	}
	if _, err := FindNeighbors(context.Background(), x, 5, NewMetric(Euclidean), NeighborSearchOptions{Algorithm: SearchExact, MemoryBudget: working, Exact: ExactOptions{BlockSize: 8}}); err != nil {
		t.Fatalf("search at predicted budget: %v", err)
	}
}

func BenchmarkNNDescentTextLike99PercentSparse(b *testing.B) {
	const rows, dimensions, perRow = 1000, 10000, 50
	values := make([]float32, rows*perRow)
	columns := make([]uint32, rows*perRow)
	offsets := make([]uint64, rows+1)
	for r := 0; r < rows; r++ {
		offsets[r] = uint64(r * perRow)
		for j := 0; j < perRow; j++ {
			values[r*perRow+j] = 1
			columns[r*perRow+j] = uint32(j*199 + r%100)
		}
	}
	offsets[rows] = uint64(len(values))
	x, _ := NewCSR(values, columns, offsets, rows, dimensions)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = NNDescent(context.Background(), x, 15, NewMetric(Cosine), NNDescentOptions{Seed: 1, MaxIterations: 5})
	}
}

func BenchmarkNeighborSearchCrossover(b *testing.B) {
	for _, rows := range []int{256, 1024, 4096} {
		data := make([]float32, rows*16)
		rng := NewRNG(99)
		for i := range data {
			data[i] = rng.Float32()
		}
		x, _ := NewDense(data, rows, 16)
		b.Run(fmt.Sprintf("exact/%d", rows), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = ExactNeighborsWithOptions(context.Background(), x, 15, NewMetric(Euclidean), ExactOptions{BlockSize: 256})
			}
		})
		b.Run(fmt.Sprintf("nndescent/%d", rows), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = NNDescent(context.Background(), x, 15, NewMetric(Euclidean), NNDescentOptions{Seed: 1})
			}
		})
	}
}
