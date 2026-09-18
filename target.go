package umap

import "math"

type TargetMetric uint8

const (
	TargetAuto TargetMetric = iota
	TargetCategorical
	TargetL1
	TargetL2
)

type TargetConfig struct {
	Neighbors int
	Metric    TargetMetric
	Weight    float32
}

func applyTarget(g Graph, target Target, rows int, cfg TargetConfig) (Graph, error) {
	if target == nil {
		return Graph{}, validationf("target cannot be nil; use NoTarget()")
	}
	if _, ok := target.(noTarget); ok || cfg.Weight == 0 {
		return g, nil
	}
	edges := append([]Edge(nil), g.Edges...)
	switch y := target.(type) {
	case Categories:
		if len(y.values) != rows {
			return Graph{}, shapef("target has %d values, need %d", len(y.values), rows)
		}
		if cfg.Metric != TargetAuto && cfg.Metric != TargetCategorical {
			return Graph{}, validationf("categorical target requires categorical metric")
		}
		for i := range edges {
			a, b := y.values[edges[i].Head], y.values[edges[i].Tail]
			if a == y.unknown || b == y.unknown {
				continue
			}
			if a != b {
				edges[i].Weight *= 1 - cfg.Weight
			}
		}
	case Continuous:
		if len(y.values) != rows {
			return Graph{}, shapef("target has %d values, need %d", len(y.values), rows)
		}
		if cfg.Metric == TargetCategorical {
			return Graph{}, validationf("continuous target does not support categorical metric")
		}
		maxDistance := float32(0)
		for _, e := range edges {
			d := float32(math.Abs(float64(y.values[e.Head] - y.values[e.Tail])))
			if cfg.Metric == TargetL2 {
				d *= d
			}
			maxDistance = max(maxDistance, d)
		}
		if maxDistance > 0 {
			for i := range edges {
				d := float32(math.Abs(float64(y.values[edges[i].Head] - y.values[edges[i].Tail])))
				if cfg.Metric == TargetL2 {
					d *= d
				}
				similarity := float32(math.Exp(-float64(d / maxDistance)))
				edges[i].Weight *= (1 - cfg.Weight) + cfg.Weight*similarity
			}
		}
	default:
		return Graph{}, validationf("unknown target type")
	}
	return Graph{Vertices: g.Vertices, Edges: edges}, nil
}
