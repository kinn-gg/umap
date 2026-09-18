// Command umap-benchmark records reproducible end-to-end Go UMAP timings.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"time"

	umap "github.com/kinn-gg/umap"
)

const seed uint64 = 42

type parameters struct {
	Rows       int     `json:"rows"`
	Dimensions int     `json:"dimensions"`
	Neighbors  int     `json:"neighbors"`
	Components int     `json:"components"`
	Density    float64 `json:"density"`
	Metric     string  `json:"metric"`
}

type benchCase struct {
	name   string
	params parameters
	matrix umap.Matrix
	sum    string
}

type allocationProbe struct {
	TimingNS          int64  `json:"fit_end_to_end_ns"`
	AllocatedBytes    uint64 `json:"go_allocated_bytes"`
	RetainedHeapBytes uint64 `json:"go_retained_heap_bytes"`
}

type caseRecord struct {
	SchemaVersion   int                `json:"schema_version"`
	Type            string             `json:"type"`
	Name            string             `json:"name"`
	Parameters      parameters         `json:"parameters"`
	InputSHA256     string             `json:"input_sha256"`
	WarmupNS        []int64            `json:"warmup_ns"`
	Runs            []map[string]int64 `json:"runs"`
	TotalMeanNS     float64            `json:"total_mean_ns"`
	TotalStdevNS    float64            `json:"total_stdev_ns"`
	AllocationProbe allocationProbe    `json:"allocation_probe"`
	PeakRSSBytes    *uint64            `json:"peak_rss_bytes"`
}

type splitMix64 struct{ state uint64 }

func (r *splitMix64) next() uint64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func value(r *splitMix64) float32 {
	return float32(float64(r.next()>>40)/float64(uint64(1)<<24)*2 - 1)
}

func makeCase(name string, p parameters) (benchCase, error) {
	rng := splitMix64{state: seed}
	h := sha256.New()
	writeFloat := func(v float32) {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		_, _ = h.Write(b[:])
	}
	if p.Density == 1 {
		data := make([]float32, p.Rows*p.Dimensions)
		for i := range data {
			data[i] = value(&rng)
			writeFloat(data[i])
		}
		x, err := umap.NewDense(data, p.Rows, p.Dimensions)
		return benchCase{name, p, x, hex.EncodeToString(h.Sum(nil))}, err
	}
	values := make([]float32, 0, int(float64(p.Rows*p.Dimensions)*p.Density)+p.Rows)
	columns := make([]uint32, 0, cap(values))
	offsets := make([]uint64, p.Rows+1)
	threshold := uint64(p.Density * float64(uint64(1)<<24))
	for row := 0; row < p.Rows; row++ {
		for column := 0; column < p.Dimensions; column++ {
			v := value(&rng)
			if rng.next()>>40 >= threshold {
				continue
			}
			values = append(values, v)
			columns = append(columns, uint32(column))
		}
		offsets[row+1] = uint64(len(values))
	}
	for _, v := range values {
		writeFloat(v)
	}
	var b [8]byte
	for _, v := range columns {
		binary.LittleEndian.PutUint64(b[:], uint64(v))
		_, _ = h.Write(b[:])
	}
	for _, v := range offsets {
		binary.LittleEndian.PutUint64(b[:], v)
		_, _ = h.Write(b[:])
	}
	x, err := umap.NewCSR(values, columns, offsets, p.Rows, p.Dimensions)
	return benchCase{name, p, x, hex.EncodeToString(h.Sum(nil))}, err
}

func cases() ([]benchCase, error) {
	base := parameters{256, 16, 15, 2, 1, "euclidean"}
	specs := []struct {
		name string
		p    parameters
	}{
		{"synthetic/rows/64", parameters{64, 16, 15, 2, 1, "euclidean"}},
		{"synthetic/rows/1024", parameters{1024, 16, 15, 2, 1, "euclidean"}},
		{"synthetic/dimensions/2", parameters{256, 2, 15, 2, 1, "euclidean"}},
		{"synthetic/dimensions/128", parameters{256, 128, 15, 2, 1, "euclidean"}},
		{"synthetic/neighbors/5", parameters{256, 16, 5, 2, 1, "euclidean"}},
		{"synthetic/neighbors/50", parameters{256, 16, 50, 2, 1, "euclidean"}},
		{"synthetic/components/2", base},
		{"synthetic/components/8", parameters{256, 16, 15, 8, 1, "euclidean"}},
		{"synthetic/density/0.05", parameters{256, 16, 15, 2, .05, "euclidean"}},
		{"synthetic/density/0.5", parameters{256, 16, 15, 2, .5, "euclidean"}},
		{"synthetic/metric/euclidean", base},
		{"synthetic/metric/cosine", parameters{256, 16, 15, 2, 1, "cosine"}},
		{"synthetic/sparse-text", parameters{1000, 5000, 15, 2, .01, "cosine"}},
	}
	out := make([]benchCase, 0, len(specs))
	for _, spec := range specs {
		c, err := makeCase(spec.name, spec.p)
		if err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, nil
}

func fit(c benchCase) error {
	cfg := umap.DefaultConfig()
	cfg.Neighbors, cfg.Components, cfg.Epochs, cfg.Init = c.params.Neighbors, c.params.Components, 100, umap.RandomInit
	cfg.Seed, cfg.Workers, cfg.Deterministic = new(uint64), 1, true
	*cfg.Seed = seed
	if c.params.Metric == "cosine" {
		cfg.Metric = umap.NewMetric(umap.Cosine)
	}
	u, err := umap.New(cfg)
	if err != nil {
		return err
	}
	_, err = u.Fit(context.Background(), c.matrix, umap.NoTarget())
	return err
}

func timedFit(c benchCase) (int64, error) {
	start := time.Now()
	err := fit(c)
	return time.Since(start).Nanoseconds(), err
}

func main() {
	repeats := flag.Int("repeats", 5, "number of measured fits")
	warmups := flag.Int("warmups", 1, "number of untimed warmup fits")
	output := flag.String("output", "", "write JSONL to this file (default stdout)")
	selected := flag.String("case", "", "run one exact case name")
	flag.Parse()
	if *repeats < 1 || *warmups < 0 {
		fmt.Fprintln(os.Stderr, "repeats must be positive and warmups non-negative")
		os.Exit(2)
	}
	all, err := cases()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var w *os.File = os.Stdout
	if *output != "" {
		w, err = os.Create(*output)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer w.Close()
	}
	enc := json.NewEncoder(w)
	build, _ := debug.ReadBuildInfo()
	version := ""
	if build != nil {
		version = build.Main.Version
	}
	_ = enc.Encode(map[string]any{"schema_version": 1, "type": "metadata", "platform": runtime.GOOS, "architecture": runtime.GOARCH, "go": runtime.Version(), "module_version": version, "seed": seed, "warmups": *warmups, "repeats": *repeats, "warmup_policy": "untimed before measured repeats"})
	matched := 0
	for _, c := range all {
		if *selected != "" && c.name != *selected {
			continue
		}
		matched++
		record := caseRecord{SchemaVersion: 1, Type: "case", Name: c.name, Parameters: c.params, InputSHA256: c.sum}
		for i := 0; i < *warmups; i++ {
			ns, err := timedFit(c)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			record.WarmupNS = append(record.WarmupNS, ns)
		}
		values := make([]float64, *repeats)
		for i := 0; i < *repeats; i++ {
			ns, err := timedFit(c)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			record.Runs = append(record.Runs, map[string]int64{"fit_end_to_end": ns, "total": ns})
			values[i] = float64(ns)
		}
		for _, v := range values {
			record.TotalMeanNS += v
		}
		record.TotalMeanNS /= float64(len(values))
		if len(values) > 1 {
			for _, v := range values {
				d := v - record.TotalMeanNS
				record.TotalStdevNS += d * d
			}
			record.TotalStdevNS = math.Sqrt(record.TotalStdevNS / float64(len(values)-1))
		}
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		ns, err := timedFit(c)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		runtime.GC()
		runtime.ReadMemStats(&after)
		record.AllocationProbe = allocationProbe{ns, after.TotalAlloc - before.TotalAlloc, after.HeapAlloc}
		if rss, ok := peakRSS(); ok {
			record.PeakRSSBytes = &rss
		}
		if err := enc.Encode(record); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	if matched == 0 {
		fmt.Fprintln(os.Stderr, "unknown case")
		os.Exit(2)
	}
}
