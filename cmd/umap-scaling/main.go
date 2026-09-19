// Command umap-scaling runs the local-only 10k-100k scaling suite.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"slices"
	"sort"
	"strings"
	"time"

	umap "github.com/kinn-gg/umap"
)

const (
	schemaVersion = 1
	seed          = uint64(42)
	searchPool    = 256
	searchIters   = 20
)

type parameters struct {
	Rows        int     `json:"rows"`
	Dimensions  int     `json:"dimensions"`
	Neighbors   int     `json:"neighbors"`
	Components  int     `json:"components"`
	Epochs      int     `json:"epochs"`
	Workers     int     `json:"workers"`
	Density     float64 `json:"density"`
	Metric      string  `json:"metric"`
	Backend     string  `json:"backend"`
	SearchPool  int     `json:"search_candidate_pool"`
	SearchIters int     `json:"search_max_iterations"`
	SearchDelta float64 `json:"search_convergence_delta"`
}

type workload struct {
	name   string
	params parameters
	matrix umap.Matrix
	sum    string
}

type sample struct {
	EndToEndNS       int64 `json:"end_to_end_ns"`
	NeighborSearchNS int64 `json:"neighbor_search_ns"`
	LayoutNS         int64 `json:"layout_ns"`
}

type budget struct {
	MedianNS int64 `json:"median_ns"`
	MADNS    int64 `json:"mad_ns"`
	LimitNS  int64 `json:"limit_ns"`
}

type allocationProbe struct {
	AllocatedBytes    uint64 `json:"allocated_bytes"`
	RetainedHeapBytes uint64 `json:"retained_heap_bytes"`
}

type caseRecord struct {
	SchemaVersion       int               `json:"schema_version"`
	Type                string            `json:"type"`
	Name                string            `json:"name"`
	Parameters          parameters        `json:"parameters"`
	Seed                uint64            `json:"seed"`
	InputSHA256         string            `json:"input_sha256"`
	Warmups             []sample          `json:"warmups"`
	Samples             []sample          `json:"samples"`
	RegressionBudgets   map[string]budget `json:"regression_budgets"`
	AllocationProbe     allocationProbe   `json:"allocation_probe"`
	PeakRSSBytes        *uint64           `json:"peak_rss_bytes"`
	RecallQueries       int               `json:"recall_queries"`
	RecallAtK           float64           `json:"recall_at_k"`
	SeededReproducible  bool              `json:"seeded_reproducible"`
	ResourceLimitReason string            `json:"resource_limit_reason,omitempty"`
}

type rng struct{ state uint64 }

func (r *rng) next() uint64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
	z = (z ^ z>>27) * 0x94d049bb133111eb
	return z ^ z>>31
}

func randomValue(r *rng) float32 {
	return float32(float64(r.next()>>40)/float64(uint64(1)<<24)*2 - 1)
}

func checksumDense(data []float32) string {
	h := sha256.New()
	var word [4]byte
	for _, value := range data {
		binary.LittleEndian.PutUint32(word[:], math.Float32bits(value))
		_, _ = h.Write(word[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func checksumCSR(values []float32, columns []uint32, offsets []uint64) string {
	h := sha256.New()
	var word [8]byte
	for _, value := range values {
		binary.LittleEndian.PutUint32(word[:4], math.Float32bits(value))
		_, _ = h.Write(word[:4])
	}
	for _, column := range columns {
		binary.LittleEndian.PutUint64(word[:], uint64(column))
		_, _ = h.Write(word[:])
	}
	for _, offset := range offsets {
		binary.LittleEndian.PutUint64(word[:], offset)
		_, _ = h.Write(word[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func makeWorkload(kind string, rows, workers, epochs int) (workload, error) {
	r := rng{state: seed}
	if kind == "dense" {
		const dimensions, clusters = 64, 128
		centers := make([]float32, clusters*dimensions)
		for i := range centers {
			centers[i] = randomValue(&r)
		}
		data := make([]float32, rows*dimensions)
		for row := range rows {
			center := row % clusters
			for column := range dimensions {
				data[row*dimensions+column] = centers[center*dimensions+column] + .05*randomValue(&r)
			}
		}
		x, err := umap.NewDense(data, rows, dimensions)
		p := parameters{Rows: rows, Dimensions: dimensions, Neighbors: 15, Components: 2, Epochs: epochs, Workers: workers, Density: 1, Metric: "euclidean", Backend: "nn_descent", SearchPool: searchPool, SearchIters: searchIters, SearchDelta: .0001}
		return workload{fmt.Sprintf("dense/%d", rows), p, x, checksumDense(data)}, err
	}
	if kind != "sparse" {
		return workload{}, fmt.Errorf("unknown workload kind %q", kind)
	}
	const dimensions, nonzeros, clusters, shared = 4096, 16, 128, 12
	values := make([]float32, 0, rows*nonzeros)
	columns := make([]uint32, 0, rows*nonzeros)
	offsets := make([]uint64, rows+1)
	for row := range rows {
		cluster := row % clusters
		seen := make(map[uint32]bool, nonzeros)
		rowColumns := make([]uint32, 0, nonzeros)
		for j := range shared {
			column := uint32((cluster*31 + j*251) % dimensions)
			seen[column] = true
			rowColumns = append(rowColumns, column)
		}
		for len(rowColumns) < nonzeros {
			column := uint32(r.next() % dimensions)
			if !seen[column] {
				seen[column] = true
				rowColumns = append(rowColumns, column)
			}
		}
		slices.Sort(rowColumns)
		for _, column := range rowColumns {
			columns = append(columns, column)
			values = append(values, 1+.05*randomValue(&r))
		}
		offsets[row+1] = uint64(len(values))
	}
	x, err := umap.NewCSR(values, columns, offsets, rows, dimensions)
	p := parameters{Rows: rows, Dimensions: dimensions, Neighbors: 15, Components: 2, Epochs: epochs, Workers: workers, Density: float64(nonzeros) / dimensions, Metric: "cosine", Backend: "nn_descent", SearchPool: searchPool, SearchIters: searchIters, SearchDelta: .0001}
	return workload{fmt.Sprintf("sparse/%d", rows), p, x, checksumCSR(values, columns, offsets)}, err
}

func fit(w workload) (sample, error) {
	cfg := umap.DefaultConfig()
	cfg.Neighbors, cfg.Components, cfg.Epochs = w.params.Neighbors, w.params.Components, w.params.Epochs
	cfg.Init, cfg.Seed, cfg.Workers = umap.RandomInit, new(uint64), w.params.Workers
	*cfg.Seed = seed
	cfg.Deterministic = true
	cfg.NeighborSearch.Algorithm = umap.SearchNNDescent
	cfg.NeighborSearch.Approximate.CandidatePoolSize = searchPool
	cfg.NeighborSearch.Approximate.MaxIterations = searchIters
	cfg.NeighborSearch.Approximate.ConvergenceDelta = .0001
	if w.params.Metric == "cosine" {
		cfg.Metric = umap.NewMetric(umap.Cosine)
	}
	var result sample
	cfg.StageTiming = func(t umap.StageTiming) {
		switch t.Stage {
		case umap.FitStageNeighborSearch:
			result.NeighborSearchNS = t.Elapsed.Nanoseconds()
		case umap.FitStageLayout:
			result.LayoutNS = t.Elapsed.Nanoseconds()
		}
	}
	u, err := umap.New(cfg)
	if err != nil {
		return result, err
	}
	started := time.Now()
	_, err = u.Fit(context.Background(), w.matrix, umap.NoTarget())
	result.EndToEndNS = time.Since(started).Nanoseconds()
	return result, err
}

func search(w workload) (umap.Neighbors, error) {
	metric := umap.NewMetric(umap.Euclidean)
	if w.params.Metric == "cosine" {
		metric = umap.NewMetric(umap.Cosine)
	}
	return umap.FindNeighbors(context.Background(), w.matrix, w.params.Neighbors, metric, umap.NeighborSearchOptions{
		Algorithm:   umap.SearchNNDescent,
		Approximate: umap.NNDescentOptions{Seed: seed, Workers: w.params.Workers, CandidatePoolSize: searchPool, MaxIterations: searchIters, ConvergenceDelta: .0001},
	})
}

func exactQuery(w workload, query, k int) []int {
	type candidate struct {
		index    int
		distance float64
	}
	rows, dimensions := w.matrix.Shape()
	metric := umap.NewMetric(umap.Euclidean)
	if w.params.Metric == "cosine" {
		metric = umap.NewMetric(umap.Cosine)
	}
	aScratch, bScratch := make([]float32, dimensions), make([]float32, dimensions)
	all := make([]candidate, rows)
	for row := range rows {
		all[row] = candidate{row, umap.MatrixDistance(metric, w.matrix, query, w.matrix, row, aScratch, bScratch)}
	}
	sort.Slice(all, func(i, j int) bool {
		return all[i].distance < all[j].distance || all[i].distance == all[j].distance && all[i].index < all[j].index
	})
	out := make([]int, k)
	for i := range k {
		out[i] = all[i].index
	}
	return out
}

func quality(w workload, queries int) (float64, bool, error) {
	a, err := search(w)
	if err != nil {
		return 0, false, err
	}
	b, err := search(w)
	if err != nil {
		return 0, false, err
	}
	reproducible := slices.Equal(a.Indices, b.Indices) && slices.Equal(a.Distances, b.Distances)
	queries = min(queries, w.params.Rows)
	hits := 0
	for i := range queries {
		query := i * w.params.Rows / queries
		exact := exactQuery(w, query, w.params.Neighbors)
		got := a.Indices[query*w.params.Neighbors : (query+1)*w.params.Neighbors]
		for _, candidate := range got {
			if slices.Contains(exact, candidate) {
				hits++
			}
		}
	}
	return float64(hits) / float64(queries*w.params.Neighbors), reproducible, nil
}

func deriveBudget(samples []int64) budget {
	ordered := append([]int64(nil), samples...)
	slices.Sort(ordered)
	median := ordered[len(ordered)/2]
	deviations := make([]int64, len(ordered))
	for i, value := range ordered {
		deviations[i] = int64(math.Abs(float64(value - median)))
	}
	slices.Sort(deviations)
	mad := deviations[len(deviations)/2]
	// A 25% floor avoids pretending a quiet three-run sample is a tight gate.
	limit := max(median+6*mad, median+median/4)
	return budget{median, mad, limit}
}

func runCase(kind string, rows, workers, epochs, warmups, repeats, queries int) (caseRecord, error) {
	w, err := makeWorkload(kind, rows, workers, epochs)
	if err != nil {
		return caseRecord{}, err
	}
	r := caseRecord{SchemaVersion: schemaVersion, Type: "case", Name: w.name, Parameters: w.params, Seed: seed, InputSHA256: w.sum, RecallQueries: min(queries, rows)}
	for range warmups {
		s, err := fit(w)
		if err != nil {
			return r, err
		}
		r.Warmups = append(r.Warmups, s)
	}
	for range repeats {
		s, err := fit(w)
		if err != nil {
			return r, err
		}
		r.Samples = append(r.Samples, s)
	}
	endToEnd, neighbors, layout := make([]int64, repeats), make([]int64, repeats), make([]int64, repeats)
	for i, s := range r.Samples {
		endToEnd[i], neighbors[i], layout[i] = s.EndToEndNS, s.NeighborSearchNS, s.LayoutNS
	}
	r.RegressionBudgets = map[string]budget{"end_to_end": deriveBudget(endToEnd), "neighbor_search": deriveBudget(neighbors), "layout": deriveBudget(layout)}
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	if _, err := fit(w); err != nil {
		return r, err
	}
	runtime.GC()
	runtime.ReadMemStats(&after)
	r.AllocationProbe = allocationProbe{AllocatedBytes: after.TotalAlloc - before.TotalAlloc, RetainedHeapBytes: after.HeapAlloc}
	r.RecallAtK, r.SeededReproducible, err = quality(w, queries)
	if rss, ok := peakRSS(); ok {
		r.PeakRSSBytes = &rss
	}
	return r, err
}

func metadata(warmups, repeats, queries, workers, epochs int) map[string]any {
	info, _ := debug.ReadBuildInfo()
	moduleVersion, revision := "", ""
	if info != nil {
		moduleVersion = info.Main.Version
		for _, setting := range info.Settings {
			if setting.Key == "vcs.revision" {
				revision = setting.Value
			}
		}
	}
	if revision == "" {
		if output, err := exec.Command("git", "rev-parse", "HEAD").Output(); err == nil {
			revision = strings.TrimSpace(string(output))
		}
	}
	return map[string]any{
		"schema_version": schemaVersion, "type": "metadata", "suite": "scaling-v1",
		"platform": runtime.GOOS, "architecture": runtime.GOARCH, "logical_cpus": runtime.NumCPU(),
		"cpu": hardwareCPU(), "go_version": runtime.Version(), "module_version": moduleVersion,
		"vcs_revision": revision, "seed": seed, "warmups": warmups, "repeats": repeats,
		"recall_queries": queries, "workers": workers, "epochs": epochs,
		"budget_policy":    "median + max(6*MAD, 25% of median)",
		"peak_rss_support": "Linux and macOS; null on other platforms",
	}
}

func main() {
	output := flag.String("output", "", "write JSONL to this file (default stdout)")
	selected := flag.String("case", "", "run one case: dense/{10000,50000,100000} or sparse/{...}")
	warmups := flag.Int("warmups", 1, "untimed warmup fits per case")
	repeats := flag.Int("repeats", 3, "measured fits per case")
	queries := flag.Int("recall-queries", 32, "deterministic exact-reference queries")
	workers := flag.Int("workers", runtime.GOMAXPROCS(0), "effective fit and search worker count")
	epochs := flag.Int("epochs", 100, "layout epochs")
	child := flag.Bool("child", false, "internal: run one isolated case")
	flag.Parse()
	if *repeats < 2 || *warmups < 0 || *queries < 1 || *workers < 1 || *epochs < 1 {
		fmt.Fprintln(os.Stderr, "repeats must be >=2; warmups non-negative; queries, workers, and epochs positive")
		os.Exit(2)
	}
	if *child {
		parts := strings.Split(*selected, "/")
		var rows int
		if len(parts) != 2 {
			fmt.Fprintln(os.Stderr, "invalid case")
			os.Exit(2)
		}
		if _, err := fmt.Sscanf(parts[1], "%d", &rows); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		r, err := runCase(parts[0], rows, *workers, *epochs, *warmups, *repeats, *queries)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := json.NewEncoder(os.Stdout).Encode(r); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	var out *os.File = os.Stdout
	var err error
	if *output != "" {
		out, err = os.Create(*output)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer out.Close()
	}
	enc := json.NewEncoder(out)
	if err := enc.Encode(metadata(*warmups, *repeats, *queries, *workers, *epochs)); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	cases := []string{"dense/10000", "dense/50000", "dense/100000", "sparse/10000", "sparse/50000", "sparse/100000"}
	if *selected != "" {
		cases = []string{*selected}
	}
	executable, err := os.Executable()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for _, name := range cases {
		args := []string{"--child", "--case", name, "--warmups", fmt.Sprint(*warmups), "--repeats", fmt.Sprint(*repeats), "--recall-queries", fmt.Sprint(*queries), "--workers", fmt.Sprint(*workers), "--epochs", fmt.Sprint(*epochs)}
		cmd := exec.Command(executable, args...)
		var stdout bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stdout, os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", name, err)
			os.Exit(1)
		}
		var record caseRecord
		if err := json.Unmarshal(stdout.Bytes(), &record); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := enc.Encode(record); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
}
