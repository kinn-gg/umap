package umap

import "math"

type MetricKind uint8

const (
	Euclidean MetricKind = iota
	SquaredEuclidean
	Manhattan
	Chebyshev
	Minkowski
	Cosine
	Correlation
	Canberra
	BrayCurtis
	Hamming
	Jaccard
	Dice
)

type Metric struct {
	Kind MetricKind
	P    float64
}

func NewMetric(kind MetricKind) Metric { return Metric{Kind: kind, P: 2} }
func MinkowskiMetric(p float64) (Metric, error) {
	if p <= 0 || math.IsNaN(p) || math.IsInf(p, 0) {
		return Metric{}, validationf("Minkowski p must be finite and positive")
	}
	return Metric{Kind: Minkowski, P: p}, nil
}

// Distance evaluates a built-in metric without allocating.
func (m Metric) Distance(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("umap: metric vector length mismatch")
	}
	sum, maxv, dot, an, bn, ma, mb := 0., 0., 0., 0., 0., 0., 0.
	if m.Kind == Correlation && len(a) > 0 {
		for i := range a {
			ma += float64(a[i])
			mb += float64(b[i])
		}
		ma /= float64(len(a))
		mb /= float64(len(a))
	}
	inter, ac, bc := 0, 0, 0
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		d := math.Abs(x - y)
		switch m.Kind {
		case Euclidean, SquaredEuclidean:
			sum += d * d
		case Manhattan:
			sum += d
		case Chebyshev:
			if d > maxv {
				maxv = d
			}
		case Minkowski:
			p := m.P
			if p == 0 {
				p = 2
			}
			sum += math.Pow(d, p)
		case Cosine:
			dot += x * y
			an += x * x
			bn += y * y
		case Correlation:
			x -= ma
			y -= mb
			dot += x * y
			an += x * x
			bn += y * y
		case Canberra:
			den := math.Abs(x) + math.Abs(y)
			if den > 0 {
				sum += d / den
			}
		case BrayCurtis:
			sum += d
			an += math.Abs(x) + math.Abs(y)
		case Hamming:
			if x != y {
				sum++
			}
		case Jaccard, Dice:
			xa, xb := x != 0, y != 0
			if xa {
				ac++
			}
			if xb {
				bc++
			}
			if xa && xb {
				inter++
			}
		}
	}
	switch m.Kind {
	case Euclidean:
		return math.Sqrt(sum)
	case SquaredEuclidean, Manhattan, Canberra:
		return sum
	case Chebyshev:
		return maxv
	case Minkowski:
		p := m.P
		if p == 0 {
			p = 2
		}
		return math.Pow(sum, 1/p)
	case Cosine, Correlation:
		if an == 0 && bn == 0 {
			return 0
		}
		if an == 0 || bn == 0 {
			return 1
		}
		return 1 - dot/math.Sqrt(an*bn)
	case BrayCurtis:
		if an == 0 {
			return 0
		}
		return sum / an
	case Hamming:
		if len(a) == 0 {
			return 0
		}
		return sum / float64(len(a))
	case Jaccard:
		den := ac + bc - inter
		if den == 0 {
			return 0
		}
		return 1 - float64(inter)/float64(den)
	case Dice:
		den := ac + bc
		if den == 0 {
			return 0
		}
		return 1 - 2*float64(inter)/float64(den)
	default:
		panic("umap: unknown metric")
	}
}

func MatrixDistance(metric Metric, a Matrix, ar int, b Matrix, br int, as, bs []float32) float64 {
	_, ac := a.Shape()
	_, bc := b.Shape()
	if ac != bc {
		panic("umap: matrix column mismatch")
	}
	if len(as) < ac || len(bs) < bc {
		panic("umap: scratch too small")
	}
	return metric.Distance(denseRow(a, ar, as[:ac]), denseRow(b, br, bs[:bc]))
}
