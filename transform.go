package umap

import (
	"context"
	"math"
)

func (m *Model) Transform(ctx context.Context, x Matrix) (*Embedding, error) {
	if x == nil {
		return nil, shapef("matrix cannot be nil")
	}
	rows, _ := x.Shape()
	if rows > int(^uint(0)>>1)/m.config.Components {
		return nil, shapef("transform output dimensions overflow")
	}
	out := make([]float32, rows*m.config.Components)
	if err := m.TransformInto(ctx, out, x); err != nil {
		return nil, err
	}
	return &Embedding{data: out, rows: rows, components: m.config.Components}, nil
}

func (m *Model) TransformInto(ctx context.Context, dst []float32, x Matrix) error {
	if m == nil || m.embedding == nil || m.training == nil {
		return validationf("model is not fitted")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if x == nil {
		return shapef("matrix cannot be nil")
	}
	rows, columns := x.Shape()
	trainingRows, trainingColumns := m.training.Shape()
	if columns != trainingColumns {
		return shapef("transform input has %d columns, need %d", columns, trainingColumns)
	}
	if rows > int(^uint(0)>>1)/m.config.Components || len(dst) != rows*m.config.Components {
		return shapef("transform destination has wrong length")
	}
	k := min(m.config.Neighbors, trainingRows)
	if k < 1 {
		return validationf("model has no training observations")
	}
	indices := make([]int, k)
	distances := make([]float64, k)
	for r := 0; r < rows; r++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		for i := range distances {
			indices[i], distances[i] = -1, math.Inf(1)
		}
		clear(dst[r*m.config.Components : (r+1)*m.config.Components])
		for j := 0; j < trainingRows; j++ {
			d := MatrixDistance(m.config.Metric, x, r, m.training, j, nil, nil)
			pos := k
			for q := 0; q < k; q++ {
				if d < distances[q] || (d == distances[q] && j < indices[q]) {
					pos = q
					break
				}
			}
			if pos < k {
				copy(indices[pos+1:], indices[pos:k-1])
				copy(distances[pos+1:], distances[pos:k-1])
				indices[pos], distances[pos] = j, d
			}
		}
		weightSum := float64(0)
		for q, d := range distances {
			w := 1 / (d + 1e-3)
			weightSum += w
			for c := 0; c < m.config.Components; c++ {
				dst[r*m.config.Components+c] += float32(w) * m.embedding.data[indices[q]*m.config.Components+c]
			}
		}
		for c := 0; c < m.config.Components; c++ {
			dst[r*m.config.Components+c] /= float32(weightSum)
		}
		// Refine the weighted-neighbor initialization while keeping the fitted
		// embedding fixed. Each query is optimized independently, making results
		// invariant to transform batch size.
		a, b := m.config.A, m.config.B
		if a == 0 {
			aa, bb := FitAB(float64(m.config.Spread), float64(m.config.MinDist))
			a, b = float32(aa), float32(bb)
		}
		epochs := m.config.Epochs / 3
		if epochs == 0 {
			epochs = 100
		}
		for epoch := 0; epoch < epochs; epoch++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			alpha := m.config.LearningRate * (1 - float32(epoch)/float32(epochs))
			for _, neighbor := range indices {
				dist2 := float32(0)
				for c := 0; c < m.config.Components; c++ {
					d := dst[r*m.config.Components+c] - m.embedding.data[neighbor*m.config.Components+c]
					dist2 += d * d
				}
				if dist2 == 0 {
					continue
				}
				coeff := -2 * a * b * float32(math.Pow(float64(dist2), float64(b-1))) / (1 + a*float32(math.Pow(float64(dist2), float64(b))))
				for c := 0; c < m.config.Components; c++ {
					d := dst[r*m.config.Components+c] - m.embedding.data[neighbor*m.config.Components+c]
					dst[r*m.config.Components+c] += clamp(coeff*d, -4, 4) * alpha
				}
			}
		}
	}
	return nil
}
