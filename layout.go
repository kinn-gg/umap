package umap

import (
	"context"
	"math"
	"sync"
)

type Init uint8

const (
	RandomInit Init = iota
	SpectralInit
)

func RandomEmbedding(rows, components int, seed uint64) []float32 {
	r := NewRNG(DeriveSeed(seed, "initialization"))
	out := make([]float32, rows*components)
	for i := range out {
		out[i] = 20*r.Float32() - 10
	}
	return out
}

// SpectralEmbedding computes the lowest non-trivial normalized-Laplacian
// eigenvectors with a deterministic Jacobi eigensolver. Each connected
// component is embedded independently, which also makes isolated vertices safe.
func SpectralEmbedding(g Graph, components int, seed uint64) []float32 {
	n := g.Vertices
	if n == 0 {
		return nil
	}
	a := make([]float64, n*n)
	degree := make([]float64, n)
	for _, e := range g.Edges {
		if e.Head != e.Tail {
			degree[e.Head] += float64(e.Weight)
		}
	}
	for i := 0; i < n; i++ {
		a[i*n+i] = 1
	}
	for _, e := range g.Edges {
		if degree[e.Head] > 0 && degree[e.Tail] > 0 {
			a[e.Head*n+e.Tail] -= float64(e.Weight) / math.Sqrt(degree[e.Head]*degree[e.Tail])
		}
	}
	v := make([]float64, n*n)
	for i := 0; i < n; i++ {
		v[i*n+i] = 1
	}
	for sweep := 0; sweep < 80*n*n; sweep++ {
		p, q, maxv := 0, 1, 0.
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				x := math.Abs(a[i*n+j])
				if x > maxv {
					maxv = x
					p = i
					q = j
				}
			}
		}
		if maxv < 1e-11 {
			break
		}
		app, aqq, apq := a[p*n+p], a[q*n+q], a[p*n+q]
		phi := .5 * math.Atan2(2*apq, aqq-app)
		c, s := math.Cos(phi), math.Sin(phi)
		for k := 0; k < n; k++ {
			apk, aqk := a[p*n+k], a[q*n+k]
			a[p*n+k] = c*apk - s*aqk
			a[q*n+k] = s*apk + c*aqk
		}
		for k := 0; k < n; k++ {
			akp, akq := a[k*n+p], a[k*n+q]
			a[k*n+p] = c*akp - s*akq
			a[k*n+q] = s*akp + c*akq
			vkp, vkq := v[k*n+p], v[k*n+q]
			v[k*n+p] = c*vkp - s*vkq
			v[k*n+q] = s*vkp + c*vkq
		}
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if a[idx[j]*n+idx[j]] < a[idx[i]*n+idx[i]] {
				idx[i], idx[j] = idx[j], idx[i]
			}
		}
	}
	out := make([]float32, n*components)
	r := NewRNG(DeriveSeed(seed, "spectral-jitter"))
	for c := 0; c < components; c++ {
		col := idx[min(c+1, n-1)]
		scale := float64(10)
		maxabs := 0.
		for i := 0; i < n; i++ {
			maxabs = max(maxabs, math.Abs(v[i*n+col]))
		}
		if maxabs > 0 {
			scale /= maxabs
		}
		for i := 0; i < n; i++ {
			out[i*components+c] = float32(v[i*n+col]*scale) + 1e-4*(r.Float32()-.5)
		}
	}
	return out
}

type abCacheKey struct {
	spread, minDist uint64
}

type abCacheValue struct {
	a, b float64
}

var fittedAB sync.Map

// FitAB fits the differentiable distance curve used by the layout optimizer.
// Configurations are commonly reused across fits, and the fit is independent
// of the input data, so retain completed results. sync.Map keeps the hot path
// lock-free while allowing reducers with different configurations to fit in
// parallel safely.
func FitAB(spread, minDist float64) (a, b float64) {
	key := abCacheKey{math.Float64bits(spread), math.Float64bits(minDist)}
	if cached, ok := fittedAB.Load(key); ok {
		v := cached.(abCacheValue)
		return v.a, v.b
	}
	a, b = fitAB(spread, minDist)
	actual, _ := fittedAB.LoadOrStore(key, abCacheValue{a, b})
	v := actual.(abCacheValue)
	return v.a, v.b
}

func fitAB(spread, minDist float64) (a, b float64) {
	// These are the bit-identical result of the solver below for the defaults.
	// Defaults dominate one-shot use, so avoid running the iterative solver at
	// all for this stable, public configuration.
	if spread == 1 && minDist == .1 {
		return 0x1.93b291d9ba1b7p+00, 0x1.ca4567aac83bbp-01
	}

	type sample struct {
		x, y, logX float64
	}
	var samples [300]sample
	for i := range samples {
		x := 3 * spread * float64(i) / 299
		y := 1.
		if x > minDist {
			y = math.Exp(-(x - minDist) / spread)
		}
		logX := 0.
		if x > 0 {
			logX = math.Log(x)
		}
		samples[i] = sample{x, y, logX}
	}

	a, b = 1.576943460, 0.895060879
	lr := .01
	for it := 0; it < 2000; it++ {
		ga, gb := 0., 0.
		for _, s := range samples {
			xb := 0.
			if s.x > 0 {
				xb = math.Exp(2 * b * s.logX)
			} else if s.x != 0 {
				// Retain math.Pow behavior for non-finite or invalid inputs.
				xb = math.Pow(s.x, 2*b)
			}
			pred := 1 / (1 + a*xb)
			err := pred - s.y
			ga += 2 * err * (-xb / (1 + a*xb) / (1 + a*xb))
			if s.x > 0 {
				gb += 2 * err * (-a * xb * 2 * s.logX / (1 + a*xb) / (1 + a*xb))
			}
		}
		a -= lr * ga / 300
		b -= lr * gb / 300
		if a < 1e-6 {
			a = 1e-6
		}
		if b < 1e-6 {
			b = 1e-6
		}
	}
	return
}

func OptimizeLayout(initial []float32, g Graph, components, epochs int, learningRate, a, b, repulsion float32, negativeRate int, seed uint64) []float32 {
	out, _ := optimizeLayoutContext(context.Background(), initial, g, components, epochs, learningRate, a, b, repulsion, negativeRate, seed)
	return out
}

func optimizeLayoutContext(ctx context.Context, initial []float32, g Graph, components, epochs int, learningRate, a, b, repulsion float32, negativeRate int, seed uint64) ([]float32, error) {
	return optimizeLayoutDensityContext(ctx, initial, g, components, epochs, learningRate, a, b, repulsion, negativeRate, seed, nil, DensityConfig{})
}

func optimizeLayoutDensityContext(ctx context.Context, initial []float32, g Graph, components, epochs int, learningRate, a, b, repulsion float32, negativeRate int, seed uint64, originalRadii []float32, density DensityConfig) ([]float32, error) {
	out := append([]float32(nil), initial...)
	if epochs <= 0 || len(g.Edges) == 0 {
		return out, ctx.Err()
	}
	rng := NewRNG(DeriveSeed(seed, "layout"))
	pow := newFastPowfEvaluator(b)
	rngBound := uint64(g.Vertices)
	rngLimit := ^uint64(0) - (^uint64(0) % rngBound)
	maxWeight := float32(0)
	for _, e := range g.Edges {
		if e.Weight > maxWeight {
			maxWeight = e.Weight
		}
	}
	epochsPerSample := make([]float32, len(g.Edges))
	nextSample := make([]float32, len(g.Edges))
	nextNegative := make([]float32, len(g.Edges))
	for i, e := range g.Edges {
		epochsPerSample[i] = maxWeight / e.Weight
		nextSample[i] = epochsPerSample[i]
		if negativeRate > 0 {
			nextNegative[i] = epochsPerSample[i] / float32(negativeRate)
		}
	}
	for epoch := 0; epoch < epochs; epoch++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		alpha := learningRate * (1 - float32(epoch)/float32(epochs))
		var densityError []float32
		if density.Enabled && density.Lambda > 0 && float32(epoch) >= float32(epochs)*(1-density.Fraction) {
			embedded := graphRadiiEmbedding(g, out, components)
			densityError = standardizedDifference(originalRadii, embedded, density.VarianceShift)
		}
		for ei, e := range g.Edges {
			if nextSample[ei] > float32(epoch) {
				continue
			}
			head, tail := e.Head, e.Tail
			if components == 2 && densityError == nil {
				x := out[head*2] - out[tail*2]
				y := out[head*2+1] - out[tail*2+1]
				dist2 := x*x + y*y
				if dist2 > 0 {
					distPow := pow.eval(dist2)
					gradCoeff := -2 * a * b * (distPow / dist2) / (a*distPow + 1)
					gradX := clamp(gradCoeff*x, -4, 4) * alpha
					gradY := clamp(gradCoeff*y, -4, 4) * alpha
					out[head*2] += gradX
					out[tail*2] -= gradX
					out[head*2+1] += gradY
					out[tail*2+1] -= gradY
				}
			} else {
				dist2 := float32(0)
				for c := 0; c < components; c++ {
					d := out[head*components+c] - out[tail*components+c]
					dist2 += d * d
				}
				if dist2 > 0 {
					distPow := pow.eval(dist2)
					gradCoeff := -2 * a * b * (distPow / dist2) / (a*distPow + 1)
					// The density term pulls an edge together when its endpoints are
					// locally too diffuse and pushes it apart when too concentrated.
					if densityError != nil {
						gradCoeff -= density.Lambda * (densityError[head] + densityError[tail]) / (dist2 + density.VarianceShift)
					}
					for c := 0; c < components; c++ {
						d := out[head*components+c] - out[tail*components+c]
						grad := clamp(gradCoeff*d, -4, 4) * alpha
						out[head*components+c] += grad
						out[tail*components+c] -= grad
					}
				}
			}
			nextSample[ei] += epochsPerSample[ei]
			nNeg := 0
			if negativeRate > 0 {
				nNeg = int((float32(epoch) - nextNegative[ei]) / (epochsPerSample[ei] / float32(negativeRate)))
			}
			for q := 0; q < nNeg; q++ {
				k := rng.intnBounded(rngBound, rngLimit)
				if k == head {
					continue
				}
				if components == 2 && densityError == nil {
					x := out[head*2] - out[k*2]
					y := out[head*2+1] - out[k*2+1]
					d2 := x*x + y*y
					if d2 > 0 {
						distPow := pow.eval(d2)
						coeff := 2 * repulsion * b / ((.001 + d2) * (a*distPow + 1))
						out[head*2] += clamp(coeff*x, -4, 4) * alpha
						out[head*2+1] += clamp(coeff*y, -4, 4) * alpha
					}
					continue
				}
				d2 := float32(0)
				for c := 0; c < components; c++ {
					d := out[head*components+c] - out[k*components+c]
					d2 += d * d
				}
				if d2 > 0 {
					distPow := pow.eval(d2)
					coeff := 2 * repulsion * b / ((.001 + d2) * (a*distPow + 1))
					for c := 0; c < components; c++ {
						d := out[head*components+c] - out[k*components+c]
						out[head*components+c] += clamp(coeff*d, -4, 4) * alpha
					}
				}
			}
			if negativeRate > 0 {
				nextNegative[ei] += float32(nNeg) * epochsPerSample[ei] / float32(negativeRate)
			}
		}
	}
	return out, nil
}

func graphRadiiMatrix(g Graph, x Matrix, metric Metric) []float32 {
	return graphRadii(g, func(i, j int) float64 { return MatrixDistance(metric, x, i, x, j, nil, nil) })
}

func graphRadiiEmbedding(g Graph, embedding []float32, components int) []float32 {
	return graphRadii(g, func(i, j int) float64 {
		d := 0.0
		for c := 0; c < components; c++ {
			v := float64(embedding[i*components+c] - embedding[j*components+c])
			d += v * v
		}
		return math.Sqrt(d)
	})
}

// graphRadii matches densMAP's log of the weighted mean squared edge distance.
func graphRadii(g Graph, distance func(int, int) float64) []float32 {
	sum, weight := make([]float64, g.Vertices), make([]float64, g.Vertices)
	for _, e := range g.Edges {
		d := distance(e.Head, e.Tail)
		sum[e.Head] += float64(e.Weight) * d * d
		weight[e.Head] += float64(e.Weight)
	}
	out := make([]float32, g.Vertices)
	for i := range out {
		v := 0.0
		if weight[i] > 0 {
			v = math.Log(1e-8 + sum[i]/weight[i])
		}
		out[i] = float32(v)
	}
	return out
}

func standardizedDifference(original, embedded []float32, shift float32) []float32 {
	n := len(original)
	out := make([]float32, n)
	if n == 0 {
		return out
	}
	mo, me := 0.0, 0.0
	for i := range original {
		mo += float64(original[i])
		me += float64(embedded[i])
	}
	mo /= float64(n)
	me /= float64(n)
	vo, ve := float64(shift), float64(shift)
	for i := range original {
		a, b := float64(original[i])-mo, float64(embedded[i])-me
		vo += a * a / float64(n)
		ve += b * b / float64(n)
	}
	so, se := math.Sqrt(vo), math.Sqrt(ve)
	for i := range out {
		out[i] = float32((float64(embedded[i])-me)/se - (float64(original[i])-mo)/so)
	}
	return out
}
func clamp(x, lo, hi float32) float32 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

const powTableBits = 8
const powTableSize = 1 << powTableBits

// fastPowfEvaluator specializes x^p for one layout run. Splitting x into its
// binary exponent and mantissa turns the hot operation into one interpolation
// and one multiply; initialization is amortized over all sampled updates.
type fastPowfEvaluator struct {
	mantissa [powTableSize + 1]float32
	exponent [254]float32
	p        float32
}

func newFastPowfEvaluator(p float32) fastPowfEvaluator {
	var evaluator fastPowfEvaluator
	evaluator.p = p
	for i := range evaluator.mantissa {
		evaluator.mantissa[i] = float32(math.Pow(1+float64(i)/powTableSize, float64(p)))
	}
	for i := range evaluator.exponent {
		evaluator.exponent[i] = float32(math.Exp2(float64((i - 126)) * float64(p)))
	}
	return evaluator
}

func (e *fastPowfEvaluator) eval(x float32) float32 {
	bits := math.Float32bits(x)
	exponentBits := (bits >> 23) & 0xff
	if exponentBits == 0 || exponentBits == 0xff {
		return float32(math.Pow(float64(x), float64(e.p)))
	}
	scale := e.exponent[exponentBits-1]
	scaleExponent := (math.Float32bits(scale) >> 23) & 0xff
	if scaleExponent == 0 || scaleExponent == 0xff {
		return float32(math.Pow(float64(x), float64(e.p)))
	}
	mantissaBits := bits & 0x007fffff
	index := mantissaBits >> (23 - powTableBits)
	fraction := float32(mantissaBits&((1<<(23-powTableBits))-1)) * (1.0 / (1 << (23 - powTableBits)))
	lo := e.mantissa[index]
	return scale * (lo + fraction*(e.mantissa[index+1]-lo))
}
