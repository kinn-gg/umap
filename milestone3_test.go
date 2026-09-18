package umap

import (
	"context"
	"errors"
	"math"
	"testing"
)

func fittedTestModel(t *testing.T) (*Model, Dense) {
	t.Helper()
	train, err := NewDense([]float32{0, 0, 0, 1, 1, 0, 1, 1, 2, 2}, 5, 2)
	if err != nil {
		t.Fatal(err)
	}
	seed := uint64(17)
	cfg := DefaultConfig()
	cfg.Neighbors, cfg.Epochs, cfg.Seed, cfg.Deterministic = 3, 8, &seed, true
	u, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	m, err := u.Fit(context.Background(), train, NoTarget())
	if err != nil {
		t.Fatal(err)
	}
	return m, train
}

func TestTransformIsDeterministicBatchIndependentAndImmutable(t *testing.T) {
	m, train := fittedTestModel(t)
	before := append([]float32(nil), m.embedding.data...)
	query, _ := NewDense([]float32{.1, .2, 1.4, 1.2}, 2, 2)
	batch, err := m.Transform(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 2; r++ {
		one, _ := NewDense(query.Row(r), 1, 2)
		got, err := m.Transform(context.Background(), one)
		if err != nil {
			t.Fatal(err)
		}
		for c := 0; c < 2; c++ {
			if got.At(0, c) != batch.At(r, c) {
				t.Fatalf("batch-dependent transform at row %d component %d", r, c)
			}
		}
	}
	if len(before) != len(m.embedding.data) {
		t.Fatal("transform changed fitted embedding shape")
	}
	for i := range before {
		if before[i] != m.embedding.data[i] {
			t.Fatal("transform mutated fitted embedding")
		}
	}
	// Fit owns its training state; caller mutation cannot affect transforms.
	baseline, _ := m.Transform(context.Background(), query)
	clear(train.data)
	after, _ := m.Transform(context.Background(), query)
	for i := range baseline.data {
		if baseline.data[i] != after.data[i] {
			t.Fatal("model retained caller-owned training storage")
		}
	}
}

func TestModelSerializationRoundTripAndCorruption(t *testing.T) {
	m, _ := fittedTestModel(t)
	query, _ := NewDense([]float32{.25, .25, 1.5, 1.5}, 2, 2)
	want, _ := m.Transform(context.Background(), query)
	encoded, err := m.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := UnmarshalModel(encoded)
	if err != nil {
		t.Fatal(err)
	}
	got, err := restored.Transform(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.data {
		if want.data[i] != got.data[i] {
			t.Fatalf("round-trip transform differs at %d", i)
		}
	}
	encoded[len(encoded)-1] ^= 1
	if _, err := UnmarshalModel(encoded); err == nil {
		t.Fatal("corrupt model was accepted")
	}
	oversized := make([]byte, len(modelMagic)+8+32)
	copy(oversized, modelMagic)
	for i := 0; i < 8; i++ {
		oversized[len(modelMagic)+i] = 0xff
	}
	if _, err := UnmarshalModel(oversized); err == nil {
		t.Fatal("oversized model declaration was accepted")
	}
}

func TestSupervisedTargetsAndCancellation(t *testing.T) {
	x, _ := NewDense([]float32{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
	seed := uint64(5)
	base := DefaultConfig()
	base.Neighbors, base.Epochs, base.Seed, base.Deterministic = 3, 4, &seed, true
	unsupervised, _ := New(base)
	_, a, err := unsupervised.FitTransform(context.Background(), x, NoTarget())
	if err != nil {
		t.Fatal(err)
	}
	zero := base
	zero.Target.Weight = 0
	u, _ := New(zero)
	y, _ := NewCategories([]int32{0, -1, 1, 1}, -1)
	_, b, err := u.FitTransform(context.Background(), x, y)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a.data {
		if a.data[i] != b.data[i] {
			t.Fatal("target weight zero did not reduce to unsupervised fit")
		}
	}
	continuous, _ := NewContinuous([]float32{0, 0.2, 0.8, 1})
	if _, _, err := unsupervised.FitTransform(context.Background(), x, continuous); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := unsupervised.FitTransform(ctx, x, NoTarget()); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation error = %v", err)
	}
}

func TestTypedValidationErrors(t *testing.T) {
	if _, err := NewDense([]float32{float32(math.NaN())}, 1, 1); err == nil {
		t.Fatal("accepted NaN")
	} else {
		var numeric *NumericError
		if !errors.As(err, &numeric) {
			t.Fatalf("NaN error type = %T", err)
		}
	}
	if _, err := NewCSR([]float32{1}, []uint32{0}, []uint64{1, 1}, 1, 1); err == nil {
		t.Fatal("accepted malformed CSR")
	} else {
		var validation *ValidationError
		if !errors.As(err, &validation) {
			t.Fatalf("CSR error type = %T", err)
		}
	}
}

func FuzzValidationAndModelDecoding(f *testing.F) {
	f.Add([]byte(modelMagic), uint8(1), uint8(1))
	f.Add([]byte{0, 1, 2}, uint8(2), uint8(3))
	f.Fuzz(func(t *testing.T, data []byte, rowsByte, columnsByte uint8) {
		_, _ = UnmarshalModel(data)
		rows, columns := int(rowsByte%16), int(columnsByte%16)+1
		values := make([]float32, len(data)/4)
		for i := range values {
			bits := uint32(data[i*4]) | uint32(data[i*4+1])<<8 | uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24
			values[i] = math.Float32frombits(bits)
		}
		_, _ = NewDense(values, rows, columns)
	})
}
