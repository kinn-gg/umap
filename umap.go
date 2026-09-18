package umap

import "context"

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
}

func DefaultConfig() Config {
	return Config{Neighbors: 15, Components: 2, Metric: NewMetric(Euclidean), LearningRate: 1, Init: SpectralInit, MinDist: .1, Spread: 1, SetOpMixRatio: 1, LocalConnectivity: 1, RepulsionStrength: 1, NegativeSampleRate: 5}
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
	return Continuous{values: append([]float32(nil), values...)}, nil
}

func New(c Config) (*UMAP, error) {
	if c.Neighbors < 2 || c.Components < 1 || c.LearningRate <= 0 || c.Spread <= 0 || c.MinDist < 0 || c.MinDist > c.Spread || c.SetOpMixRatio < 0 || c.SetOpMixRatio > 1 || c.LocalConnectivity < 0 || c.NegativeSampleRate < 0 {
		return nil, validationf("invalid UMAP configuration")
	}
	if (c.A == 0) != (c.B == 0) {
		return nil, validationf("A and B must be specified together")
	}
	if c.Deterministic && c.Workers > 1 {
		return nil, validationf("deterministic mode requires at most one worker")
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
	embedding *Embedding
	graph     Graph
	seed      uint64
}

func (m *Model) Embedding() *Embedding { return m.embedding }
func (u *UMAP) Fit(ctx context.Context, x Matrix, _ ...any) (*Model, error) {
	m, _, err := u.FitTransform(ctx, x)
	return m, err
}
func (u *UMAP) FitTransform(ctx context.Context, x Matrix, _ ...any) (*Model, *Embedding, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	rows, _ := x.Shape()
	if rows < 2 {
		return nil, nil, shapef("UMAP needs at least two rows")
	}
	cfg := u.config
	k := min(cfg.Neighbors, rows-1)
	knn, err := ExactNeighbors(x, k, cfg.Metric)
	if err != nil {
		return nil, nil, err
	}
	rho, sigma := SmoothKNN(knn, cfg.LocalConnectivity, 1)
	graph := FuzzyGraph(knn, rho, sigma, cfg.SetOpMixRatio)
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	seed := uint64(0)
	if cfg.Seed != nil {
		seed = *cfg.Seed
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
	data := OptimizeLayout(init, graph, cfg.Components, epochs, cfg.LearningRate, a, b, cfg.RepulsionStrength, cfg.NegativeSampleRate, seed)
	e := &Embedding{data, rows, cfg.Components}
	m := &Model{e, graph, seed}
	return m, e, nil
}
