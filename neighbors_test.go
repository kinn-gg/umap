package umap

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"testing"
)

func TestParallelNeighborSearchReproducible(t *testing.T) {
	const rows, dimensions, k = 128, 12, 8
	data := make([]float32, rows*dimensions)
	rng := NewRNG(77)
	for i := range data {
		data[i] = rng.Float32()
	}
	x, _ := NewDense(data, rows, dimensions)
	var reference Neighbors
	for _, workers := range []int{1, 2, 4, 8, runtime.GOMAXPROCS(0)} {
		got, err := ExactNeighborsWithOptions(context.Background(), x, k, NewMetric(Euclidean), ExactOptions{Workers: workers, BlockSize: 31})
		if err != nil {
			t.Fatal(err)
		}
		if reference.Indices == nil {
			reference = got
			continue
		}
		for i := range got.Indices {
			if got.Indices[i] != reference.Indices[i] || got.Distances[i] != reference.Distances[i] {
				t.Fatalf("workers=%d differs at %d", workers, i)
			}
		}
	}
	var approximate Neighbors
	for _, workers := range []int{1, 2, 4, 8} {
		got, err := NNDescent(context.Background(), x, k, NewMetric(Euclidean), NNDescentOptions{Seed: 42, Workers: workers, MaxIterations: 8})
		if err != nil {
			t.Fatal(err)
		}
		if approximate.Indices == nil {
			approximate = got
			continue
		}
		for i := range got.Indices {
			if got.Indices[i] != approximate.Indices[i] || got.Distances[i] != approximate.Distances[i] {
				t.Fatalf("NN-descent workers=%d differs at %d", workers, i)
			}
		}
	}
}

func TestExactWorkerSelection(t *testing.T) {
	small, _ := NewDense(make([]float32, 64*8), 64, 8)
	large, _ := NewDense(make([]float32, 512*32), 512, 32)
	sparse, _ := NewCSR(nil, nil, make([]uint64, 513), 512, 32)
	if got := exactWorkers(small, NewMetric(Euclidean), 0); got != 1 {
		t.Fatalf("small automatic worker count = %d, want 1", got)
	}
	if got := exactWorkers(large, NewMetric(Euclidean), 0); runtime.GOMAXPROCS(0) >= 3 && got == 1 {
		t.Fatalf("large automatic worker count = %d, want parallel", got)
	}
	if got := exactWorkers(sparse, NewMetric(Cosine), 0); got != 1 {
		t.Fatalf("sparse cosine automatic worker count = %d, want 1", got)
	}
	if got := exactWorkers(small, NewMetric(Euclidean), 4); got != 4 {
		t.Fatalf("explicit worker count = %d, want 4", got)
	}
}

func TestHotPathAllocationGate(t *testing.T) {
	a, b := make([]float32, 128), make([]float32, 128)
	for _, kind := range []MetricKind{Euclidean, SquaredEuclidean, Cosine} {
		metric := NewMetric(kind)
		if got := testing.AllocsPerRun(1000, func() { _ = metric.Distance(a, b) }); got != 0 {
			t.Fatalf("metric %d hot path allocates %.2f objects/run", kind, got)
		}
	}
}

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

func BenchmarkExactSparseTextCosine(b *testing.B) {
	const rows, dimensions, perRow = 1000, 5000, 50
	values := make([]float32, rows*perRow)
	columns := make([]uint32, rows*perRow)
	offsets := make([]uint64, rows+1)
	for r := 0; r < rows; r++ {
		offsets[r] = uint64(r * perRow)
		for j := 0; j < perRow; j++ {
			values[r*perRow+j] = float32((j%7)+1) / 7
			columns[r*perRow+j] = uint32(j*97 + r%97)
		}
	}
	offsets[rows] = uint64(len(values))
	x, _ := NewCSR(values, columns, offsets, rows, dimensions)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _ = ExactNeighborsWithOptions(context.Background(), x, 15, NewMetric(Cosine), ExactOptions{Workers: 1})
	}
}

func TestSparseCosineExactMatchesParallelFullScan(t *testing.T) {
	values := []float32{1, 2, -1, 3, 4, 2}
	columns := []uint32{0, 4, 1, 4, 2, 5}
	offsets := []uint64{0, 2, 2, 4, 6}
	x, err := NewCSR(values, columns, offsets, 4, 6)
	if err != nil {
		t.Fatal(err)
	}
	one, err := ExactNeighborsWithOptions(context.Background(), x, 3, NewMetric(Cosine), ExactOptions{Workers: 1, BlockSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	parallel, err := ExactNeighborsWithOptions(context.Background(), x, 3, NewMetric(Cosine), ExactOptions{Workers: 2, BlockSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(one.Indices, parallel.Indices) || !slices.Equal(one.Distances, parallel.Distances) {
		t.Fatalf("single-worker triangular search differs from full scan\none: %#v\nfull: %#v", one, parallel)
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

func BenchmarkExactNeighborWorkers(b *testing.B) {
	type benchmarkCase struct {
		name       string
		rows, dims int
		metric     MetricKind
		sparse     bool
	}
	cases := []benchmarkCase{
		{"dense-euclidean/128x16", 128, 16, Euclidean, false},
		{"dense-euclidean/512x32", 512, 32, Euclidean, false},
		{"dense-euclidean/1024x128", 1024, 128, Euclidean, false},
		{"dense-cosine/128x16", 128, 16, Cosine, false},
		{"dense-cosine/512x32", 512, 32, Cosine, false},
		{"dense-cosine/1024x128", 1024, 128, Cosine, false},
		{"sparse-cosine/128x128", 128, 128, Cosine, true},
		{"sparse-cosine/512x1024", 512, 1024, Cosine, true},
	}
	for _, c := range cases {
		var x Matrix
		if c.sparse {
			perRow := min(8, c.dims)
			values := make([]float32, c.rows*perRow)
			columns := make([]uint32, len(values))
			offsets := make([]uint64, c.rows+1)
			for row := range c.rows {
				offsets[row] = uint64(row * perRow)
				for j := range perRow {
					values[row*perRow+j] = float32(j+1) / float32(perRow)
					stride := c.dims / perRow
					columns[row*perRow+j] = uint32(j*stride + row%stride)
				}
			}
			offsets[c.rows] = uint64(len(values))
			x, _ = NewCSR(values, columns, offsets, c.rows, c.dims)
		} else {
			data := make([]float32, c.rows*c.dims)
			rng := NewRNG(99)
			for i := range data {
				data[i] = rng.Float32()
			}
			x, _ = NewDense(data, c.rows, c.dims)
		}
		for _, workers := range []int{1, 2, 4, 8, 0} {
			label := fmt.Sprintf("workers-%d", workers)
			if workers == 0 {
				label = "workers-auto"
			}
			b.Run(c.name+"/"+label, func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					_, _ = ExactNeighborsWithOptions(context.Background(), x, 15, NewMetric(c.metric), ExactOptions{Workers: workers})
				}
			})
		}
	}
}
