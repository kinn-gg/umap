package parity

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"
)

type benchShape struct {
	rows, dimensions, neighbors, components int
	density                                 float64
	metric                                  string
}

func benchmarkArtifact(s benchShape) *Artifact {
	data := make([]float64, s.rows*s.dimensions)
	for i := range data {
		data[i] = math.Sin(float64(i)*.17) + math.Cos(float64(i)*.03)
	}
	input := Matrix{Rows: s.rows, Columns: s.dimensions, Data: data}
	if s.density < 1 {
		values, indices, indptr := make([]float64, 0), make([]int, 0), make([]int, s.rows+1)
		stride := int(math.Round(1 / s.density))
		for r := 0; r < s.rows; r++ {
			for c := 0; c < s.dimensions; c++ {
				i := r*s.dimensions + c
				if i%stride == 0 {
					values = append(values, data[i])
					indices = append(indices, c)
				}
			}
			indptr[r+1] = len(values)
		}
		input = Matrix{Rows: s.rows, Columns: s.dimensions, Values: values, Indices: indices, Indptr: indptr}
	}
	knnI := make([]int, s.rows*s.neighbors)
	knnD := make([]float64, len(knnI))
	for r := 0; r < s.rows; r++ {
		for k := 0; k < s.neighbors; k++ {
			p := r*s.neighbors + k
			knnI[p] = (r + k) % s.rows
			knnD[p] = float64(k)
		}
	}
	rho, sigma := make([]float64, s.rows), make([]float64, s.rows)
	for i := range rho {
		rho[i] = 1
		sigma[i] = .5
	}
	edges := make([]Edge, 0, s.rows*s.neighbors)
	for r := 0; r < s.rows; r++ {
		for j := 1; j <= s.neighbors; j++ {
			edges = append(edges, Edge{Head: r, Tail: (r + j) % s.rows, Weight: 1 / float64(j)})
		}
	}
	coords := make([]float64, s.rows*s.components)
	for i := range coords {
		coords[i] = math.Sin(float64(i) * .11)
	}
	return &Artifact{SchemaVersion: 1, Name: "benchmark", Suite: "benchmark", Parameters: Parameters{Neighbors: s.neighbors, Components: s.components, Metric: s.metric, MinDist: .1, Spread: 1, LocalConnectivity: 1, Epochs: 100}, Input: input, KNN: KNN{Indices: knnI, Distances: knnD}, SmoothKNN: SmoothKNN{Rho: rho, Sigma: sigma}, Graph: edges, Initialization: coords, Embedding: coords}
}

func BenchmarkCompareScaling(b *testing.B) {
	base := benchShape{rows: 256, dimensions: 16, neighbors: 15, components: 2, density: 1, metric: "euclidean"}
	cases := []struct {
		name  string
		shape benchShape
	}{
		{"rows/64", benchShape{64, 16, 15, 2, 1, "euclidean"}}, {"rows/1024", benchShape{1024, 16, 15, 2, 1, "euclidean"}},
		{"dimensions/2", benchShape{256, 2, 15, 2, 1, "euclidean"}}, {"dimensions/128", benchShape{256, 128, 15, 2, 1, "euclidean"}},
		{"neighbors/5", benchShape{256, 16, 5, 2, 1, "euclidean"}}, {"neighbors/50", benchShape{256, 16, 50, 2, 1, "euclidean"}},
		{"components/2", base}, {"components/8", benchShape{256, 16, 15, 8, 1, "euclidean"}},
		{"density/0.05", benchShape{256, 16, 15, 2, .05, "euclidean"}}, {"density/0.5", benchShape{256, 16, 15, 2, .5, "euclidean"}},
		{"metric/euclidean", base}, {"metric/cosine", benchShape{256, 16, 15, 2, 1, "cosine"}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			ref := benchmarkArtifact(tc.shape)
			got := benchmarkArtifact(tc.shape)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				result := Compare(ref, got, DefaultThresholds)
				if !result.Passed {
					b.Fatal(result)
				}
			}
		})
	}
}

func BenchmarkCorpusEndToEnd(b *testing.B) {
	paths, err := filepath.Glob(filepath.Join("..", "..", "parity", "fixtures", "*.json"))
	if err != nil {
		b.Fatal(err)
	}
	for _, path := range paths {
		if filepath.Base(path) == "manifest.json" {
			continue
		}
		b.Run(fmt.Sprintf("fixture/%s", filepath.Base(path)), func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				ref, err := Load(path)
				if err != nil {
					b.Fatal(err)
				}
				got, err := Load(path)
				if err != nil {
					b.Fatal(err)
				}
				if r := Compare(ref, got, DefaultThresholds); !r.Passed {
					b.Fatal(r)
				}
			}
		})
	}
}
