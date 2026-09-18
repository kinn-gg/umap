package umap

import (
	"context"
	"math"
	"testing"
)

func densityFixture(t *testing.T, density DensityConfig) (*Model, *Embedding) {
	t.Helper()
	data := make([]float32, 48)
	for i := 0; i < 12; i++ {
		s := float32(1)
		if i >= 6 {
			s = 6
		}
		data[4*i], data[4*i+1] = s*float32(i%3), s*float32((i/3)%2)
	}
	x, err := NewDense(data, 12, 4)
	if err != nil {
		t.Fatal(err)
	}
	seed := uint64(31)
	cfg := DefaultConfig()
	cfg.Neighbors, cfg.Epochs, cfg.Init, cfg.Seed = 4, 30, RandomInit, &seed
	cfg.Deterministic, cfg.Density = true, density
	u, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	m, e, err := u.FitTransform(context.Background(), x, NoTarget())
	if err != nil {
		t.Fatal(err)
	}
	return m, e
}

func TestDensityOutputsAndSerialization(t *testing.T) {
	m, _ := densityFixture(t, DensityConfig{Output: true, Lambda: 2, Fraction: .3, VarianceShift: .1})
	ro, ok := m.OriginalRadii()
	if !ok || ro.Len() != 12 {
		t.Fatal("missing original radii")
	}
	re, ok := m.EmbeddingRadii()
	if !ok || re.Len() != 12 {
		t.Fatal("missing embedding radii")
	}
	for i := 0; i < ro.Len(); i++ {
		if math.IsNaN(float64(ro.At(i))) || math.IsInf(float64(ro.At(i)), 0) || math.IsNaN(float64(re.At(i))) || math.IsInf(float64(re.At(i)), 0) {
			t.Fatal("non-finite density radius")
		}
	}
	b, err := m.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := UnmarshalModel(b)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := restored.EmbeddingRadii()
	if !ok || got.At(3) != re.At(3) {
		t.Fatal("density radii did not survive serialization")
	}
}

func TestOriginalRadiiMatchPythonReference(t *testing.T) {
	x, _ := NewDense([]float32{0, 0, 0, 1, 2, 0, 5, 1, 9, 0}, 5, 2)
	seed := uint64(9)
	cfg := DefaultConfig()
	cfg.Neighbors, cfg.Epochs, cfg.Init, cfg.Seed = 3, 10, RandomInit, &seed
	cfg.Density.Output = true
	u, _ := New(cfg)
	m, err := u.Fit(context.Background(), x, NoTarget())
	if err != nil {
		t.Fatal(err)
	}
	got, _ := m.OriginalRadii()
	want := []float32{.91629076, 1.0337199, 2.6184173, 2.6026897, 3.3607304}
	for i := range want {
		if math.Abs(float64(got.At(i)-want[i])) > 2e-5 {
			t.Fatalf("original radius %d = %g, want %g", i, got.At(i), want[i])
		}
	}
}

func TestZeroDensityLambdaMatchesOrdinaryUMAP(t *testing.T) {
	plain, a := densityFixture(t, DefaultConfig().Density)
	if _, ok := plain.OriginalRadii(); ok {
		t.Fatal("ordinary fit allocated density outputs")
	}
	_, b := densityFixture(t, DensityConfig{Enabled: true, Lambda: 0, Fraction: .3, VarianceShift: .1})
	for i := range a.data {
		if a.data[i] != b.data[i] {
			t.Fatalf("lambda zero differs at coordinate %d", i)
		}
	}
}

func TestDensMAPImprovesDensityCorrelation(t *testing.T) {
	plain, _ := densityFixture(t, DensityConfig{Output: true, Lambda: 2, Fraction: .3, VarianceShift: .1})
	dense, _ := densityFixture(t, DensityConfig{Enabled: true, Output: true, Lambda: 2, Fraction: .3, VarianceShift: .1})
	correlation := func(m *Model) float64 {
		a, _ := m.OriginalRadii()
		b, _ := m.EmbeddingRadii()
		ma, mb := 0.0, 0.0
		for i := 0; i < a.Len(); i++ {
			ma += float64(a.At(i))
			mb += float64(b.At(i))
		}
		ma /= float64(a.Len())
		mb /= float64(a.Len())
		num, da, db := 0.0, 0.0, 0.0
		for i := 0; i < a.Len(); i++ {
			x, y := float64(a.At(i))-ma, float64(b.At(i))-mb
			num += x * y
			da += x * x
			db += y * y
		}
		return num / math.Sqrt(da*db)
	}
	ordinary, preserved := correlation(plain), correlation(dense)
	if preserved <= ordinary {
		t.Fatalf("density correlation did not improve: ordinary=%g densMAP=%g", ordinary, preserved)
	}
}

func TestDensityConfigValidation(t *testing.T) {
	for _, d := range []DensityConfig{{Lambda: -1}, {Fraction: -1}, {Fraction: 1.1}, {VarianceShift: -1}} {
		c := DefaultConfig()
		c.Density = d
		if _, err := New(c); err == nil {
			t.Fatalf("accepted invalid density config %+v", d)
		}
	}
}

func BenchmarkDensityOverhead(b *testing.B) {
	data := make([]float32, 128*8)
	for i := range data {
		data[i] = float32((i*17)%101) / 101
	}
	x, _ := NewDense(data, 128, 8)
	seed := uint64(1)
	for _, tc := range []struct {
		name    string
		density DensityConfig
	}{{"disabled", DefaultConfig().Density}, {"enabled", DensityConfig{Enabled: true, Lambda: 2, Fraction: .3, VarianceShift: .1}}} {
		b.Run(tc.name, func(b *testing.B) {
			cfg := DefaultConfig()
			cfg.Neighbors, cfg.Epochs, cfg.Seed, cfg.Density = 10, 30, &seed, tc.density
			u, _ := New(cfg)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := u.Fit(context.Background(), x, NoTarget()); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
