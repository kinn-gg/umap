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
	switch x := a.(type) {
	case Dense:
		switch y := b.(type) {
		case Dense:
			return metric.Distance(x.Row(ar), y.Row(br))
		case CSR:
			cs, vs := y.Row(br)
			return metric.sparseDense(cs, vs, x.Row(ar), true)
		}
	case CSR:
		acs, avs := x.Row(ar)
		switch y := b.(type) {
		case Dense:
			return metric.sparseDense(acs, avs, y.Row(br), false)
		case CSR:
			bcs, bvs := y.Row(br)
			return metric.sparseSparse(acs, avs, bcs, bvs, ac)
		}
	}
	panic("umap: unsupported matrix implementation")
}

// sparseDense evaluates a sparse row against a dense row without materializing
// the sparse input. swapped reports that the sparse row is the second operand;
// all built-in metrics are symmetric, but retaining the distinction makes this
// helper safe to extend alongside the metric set.
func (m Metric) sparseDense(cs []uint32, vs, dense []float32, swapped bool) float64 {
	_ = swapped
	// Binary metrics and correlation depend on implicit zero dimensions. A
	// single merge-style loop over the dense row handles those semantics while
	// remaining allocation-free.
	sum, maxv, dot, an, bn, ma, mb := 0., 0., 0., 0., 0., 0., 0.
	if m.Kind == Correlation && len(dense) > 0 {
		for _, v := range vs {
			ma += float64(v)
		}
		for _, v := range dense {
			mb += float64(v)
		}
		ma /= float64(len(dense))
		mb /= float64(len(dense))
	}
	inter, ac, bc, p := 0, 0, 0, 0
	for i, dv := range dense {
		sv := float32(0)
		if p < len(cs) && int(cs[p]) == i {
			sv = vs[p]
			p++
		}
		x, y := float64(sv), float64(dv)
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
			power := m.P
			if power == 0 {
				power = 2
			}
			sum += math.Pow(d, power)
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
	return finishDistance(m, len(dense), sum, maxv, dot, an, bn, inter, ac, bc)
}

func (m Metric) sparseSparse(ac []uint32, av []float32, bc []uint32, bv []float32, dimensions int) float64 {
	sum, maxv, dot, an, bn, ma, mb := 0., 0., 0., 0., 0., 0., 0.
	if m.Kind == Correlation && dimensions > 0 {
		for _, v := range av {
			ma += float64(v)
		}
		for _, v := range bv {
			mb += float64(v)
		}
		ma /= float64(dimensions)
		mb /= float64(dimensions)
	}
	inter, anc, bnc, i, j := 0, 0, 0, 0, 0
	// Correlation has a non-zero contribution at dimensions implicit in both
	// rows, so account for their count after visiting the union of indices.
	visited := 0
	for i < len(ac) || j < len(bc) {
		x, y := 0., 0.
		switch {
		case j >= len(bc) || (i < len(ac) && ac[i] < bc[j]):
			x = float64(av[i])
			i++
		case i >= len(ac) || bc[j] < ac[i]:
			y = float64(bv[j])
			j++
		default:
			x = float64(av[i])
			y = float64(bv[j])
			i++
			j++
		}
		visited++
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
			power := m.P
			if power == 0 {
				power = 2
			}
			sum += math.Pow(d, power)
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
				anc++
			}
			if xb {
				bnc++
			}
			if xa && xb {
				inter++
			}
		}
	}
	if m.Kind == Correlation && dimensions > visited {
		z := float64(dimensions - visited)
		dot += z * ma * mb
		an += z * ma * ma
		bn += z * mb * mb
	}
	return finishDistance(m, dimensions, sum, maxv, dot, an, bn, inter, anc, bnc)
}

func finishDistance(m Metric, dimensions int, sum, maxv, dot, an, bn float64, inter, ac, bc int) float64 {
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
		if dimensions == 0 {
			return 0
		}
		return sum / float64(dimensions)
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
