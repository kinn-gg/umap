package umap

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"math"
	"time"
)

// FitStage identifies an observable stage of an end-to-end fit.
type FitStage string

const (
	FitStageNeighborSearch FitStage = "neighbor_search"
	FitStageLayout         FitStage = "layout"
)

// StageTiming reports elapsed wall time for a completed fit stage. The hook is
// intended for observability and benchmarking; it is called synchronously.
type StageTiming struct {
	Stage   FitStage
	Elapsed time.Duration
}

type Config struct {
	Neighbors, Components, Epochs                                        int
	Metric                                                               Metric
	LearningRate                                                         float32
	Init                                                                 Init
	MinDist, Spread, SetOpMixRatio, LocalConnectivity, RepulsionStrength float32
	NegativeSampleRate                                                   int
	A, B                                                                 float32
	Seed                                                                 *uint64
	Deterministic                                                        bool
	Workers                                                              int
	NeighborSearch                                                       NeighborSearchOptions
	MemoryBudget                                                         int64
	Progress                                                             ProgressFunc
	StageTiming                                                          func(StageTiming)
	TransformSeed                                                        uint64
	Target                                                               TargetConfig
	Density                                                              DensityConfig
}

// DensityConfig controls densMAP and density-radius output. Enabled adds the
// density-correlation objective; Output records radii without changing layout.
type DensityConfig struct {
	Enabled, Output                 bool
	Lambda, Fraction, VarianceShift float32
}

func DefaultConfig() Config {
	return Config{Neighbors: 15, Components: 2, Metric: NewMetric(Euclidean), LearningRate: 1, Init: SpectralInit, MinDist: .1, Spread: 1, SetOpMixRatio: 1, LocalConnectivity: 1, RepulsionStrength: 1, NegativeSampleRate: 5, TransformSeed: 42, Target: TargetConfig{Weight: .5}, Density: DensityConfig{Lambda: 2, Fraction: .3, VarianceShift: .1}}
}

type UMAP struct{ config Config }

type Target interface{ target() }
type noTarget struct{}

func (noTarget) target() {}
func NoTarget() Target   { return noTarget{} }

type Categories struct {
	values  []int32
	unknown int32
}

func (Categories) target() {}
func NewCategories(values []int32, unknown int32) (Categories, error) {
	if len(values) == 0 {
		return Categories{}, shapef("categories cannot be empty")
	}
	return Categories{values: append([]int32(nil), values...), unknown: unknown}, nil
}

type Continuous struct{ values []float32 }

func (Continuous) target() {}
func NewContinuous(values []float32) (Continuous, error) {
	if len(values) == 0 {
		return Continuous{}, shapef("continuous target cannot be empty")
	}
	for _, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return Continuous{}, numericf("continuous target contains non-finite value")
		}
	}
	return Continuous{values: append([]float32(nil), values...)}, nil
}

func New(c Config) (*UMAP, error) {
	finite := func(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
	if c.Neighbors < 2 || c.Components < 1 || c.Epochs < 0 || c.Workers < 0 || c.LearningRate <= 0 || c.Spread <= 0 || c.MinDist < 0 || c.MinDist > c.Spread || c.SetOpMixRatio < 0 || c.SetOpMixRatio > 1 || c.LocalConnectivity < 0 || c.RepulsionStrength < 0 || c.NegativeSampleRate < 0 || c.MemoryBudget < 0 || c.Target.Weight < 0 || c.Target.Weight > 1 || c.Target.Neighbors < 0 || c.Density.Lambda < 0 || c.Density.Fraction < 0 || c.Density.Fraction > 1 || c.Density.VarianceShift < 0 || !finite(c.LearningRate) || !finite(c.Spread) || !finite(c.MinDist) || !finite(c.SetOpMixRatio) || !finite(c.LocalConnectivity) || !finite(c.RepulsionStrength) || !finite(c.Target.Weight) || !finite(c.Density.Lambda) || !finite(c.Density.Fraction) || !finite(c.Density.VarianceShift) || !finite(c.A) || !finite(c.B) {
		return nil, validationf("invalid UMAP configuration")
	}
	if (c.A == 0) != (c.B == 0) {
		return nil, validationf("A and B must be specified together")
	}
	if c.Metric.Kind > Dice || c.Init > SpectralInit || c.Target.Metric > TargetL2 || c.NeighborSearch.Algorithm > SearchNNDescent {
		return nil, validationf("configuration contains an unknown enum value")
	}
	if c.Metric.Kind == Minkowski && (c.Metric.P <= 0 || math.IsNaN(c.Metric.P) || math.IsInf(c.Metric.P, 0)) {
		return nil, validationf("Minkowski p must be finite and positive")
	}
	return &UMAP{c}, nil
}

type Embedding struct {
	data             []float32
	rows, components int
}

func (e *Embedding) Shape() (int, int)   { return e.rows, e.components }
func (e *Embedding) At(r, c int) float32 { return e.data[r*e.components+c] }
func (e *Embedding) CopyTo(dst []float32) error {
	if len(dst) != len(e.data) {
		return shapef("embedding destination has wrong length")
	}
	copy(dst, e.data)
	return nil
}

type Model struct {
	embedding                     *Embedding
	graph                         Graph
	seed                          uint64
	config                        Config
	training                      Matrix
	originalRadii, embeddingRadii *Vector
}

// Vector is an immutable learned one-dimensional output.
type Vector struct{ data []float32 }

func (v *Vector) Len() int {
	if v == nil {
		return 0
	}
	return len(v.data)
}
func (v *Vector) At(i int) float32 { return v.data[i] }
func (v *Vector) CopyTo(dst []float32) error {
	if v == nil || len(dst) != len(v.data) {
		return shapef("vector destination has wrong length")
	}
	copy(dst, v.data)
	return nil
}
func (m *Model) OriginalRadii() (*Vector, bool) {
	if m == nil || m.originalRadii == nil {
		return nil, false
	}
	return m.originalRadii, true
}
func (m *Model) EmbeddingRadii() (*Vector, bool) {
	if m == nil || m.embeddingRadii == nil {
		return nil, false
	}
	return m.embeddingRadii, true
}

func (m *Model) Embedding() *Embedding { return m.embedding }
func (m *Model) Graph() Graph {
	if m == nil {
		return Graph{}
	}
	return Graph{Vertices: m.graph.Vertices, Edges: append([]Edge(nil), m.graph.Edges...)}
}
func (u *UMAP) Fit(ctx context.Context, x Matrix, y Target) (*Model, error) {
	m, _, err := u.FitTransform(ctx, x, y)
	return m, err
}
func (u *UMAP) FitTransform(ctx context.Context, x Matrix, y Target) (*Model, *Embedding, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if x == nil {
		return nil, nil, shapef("matrix cannot be nil")
	}
	rows, _ := x.Shape()
	if rows < 2 {
		return nil, nil, shapef("UMAP needs at least two rows")
	}
	cfg := u.config
	k := min(cfg.Neighbors, rows-1)
	seed := randomSeed()
	if cfg.Seed != nil {
		seed = *cfg.Seed
	}
	search := cfg.NeighborSearch
	if cfg.MemoryBudget > 0 {
		search.Exact.MemoryBudget = cfg.MemoryBudget
		search.MemoryBudget = cfg.MemoryBudget
	}
	if search.Exact.Progress == nil {
		search.Exact.Progress = cfg.Progress
	}
	if search.Approximate.Progress == nil {
		search.Approximate.Progress = cfg.Progress
	}
	if cfg.Seed != nil {
		search.Approximate.Seed = seed
	}
	search.Exact.Workers = cfg.Workers
	search.Approximate.Workers = cfg.Workers
	stageStart := time.Now()
	knn, err := FindNeighbors(ctx, x, k, cfg.Metric, search)
	if err != nil {
		return nil, nil, err
	}
	if cfg.StageTiming != nil {
		cfg.StageTiming(StageTiming{Stage: FitStageNeighborSearch, Elapsed: time.Since(stageStart)})
	}
	rho, sigma := SmoothKNN(knn, cfg.LocalConnectivity, 1)
	graph := FuzzyGraph(knn, rho, sigma, cfg.SetOpMixRatio)
	graph, err = applyTarget(graph, y, rows, cfg.Target)
	if err != nil {
		return nil, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	var init []float32
	if cfg.Init == RandomInit {
		init = RandomEmbedding(rows, cfg.Components, seed)
	} else {
		init = SpectralEmbedding(graph, cfg.Components, seed)
	}
	a, b := cfg.A, cfg.B
	if a == 0 {
		aa, bb := FitAB(float64(cfg.Spread), float64(cfg.MinDist))
		a, b = float32(aa), float32(bb)
	}
	epochs := cfg.Epochs
	if epochs == 0 {
		epochs = 200
		if rows <= 10000 {
			epochs = 500
		}
	}
	var originalRadii []float32
	if cfg.Density.Enabled || cfg.Density.Output {
		originalRadii = graphRadiiMatrix(graph, x, cfg.Metric)
	}
	stageStart = time.Now()
	data, err := optimizeLayoutDensityContext(ctx, init, graph, cfg.Components, epochs, cfg.LearningRate, a, b, cfg.RepulsionStrength, cfg.NegativeSampleRate, seed, originalRadii, cfg.Density)
	if err != nil {
		return nil, nil, err
	}
	if cfg.StageTiming != nil {
		cfg.StageTiming(StageTiming{Stage: FitStageLayout, Elapsed: time.Since(stageStart)})
	}
	e := &Embedding{data, rows, cfg.Components}
	training, err := cloneMatrix(x)
	if err != nil {
		return nil, nil, err
	}
	m := &Model{embedding: e, graph: graph, seed: seed, config: cfg, training: training}
	if cfg.Density.Enabled || cfg.Density.Output {
		m.originalRadii = &Vector{originalRadii}
		m.embeddingRadii = &Vector{graphRadiiEmbedding(graph, data, cfg.Components)}
	}
	return m, e, nil
}

func randomSeed() uint64 {
	var b [8]byte
	if _, err := rand.Read(b[:]); err == nil {
		return binary.LittleEndian.Uint64(b[:])
	}
	return 0
}
