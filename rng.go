package umap

import "encoding/binary"

// RNG is a small, architecture-independent SplitMix64 generator.
type RNG struct{ state uint64 }

func NewRNG(seed uint64) *RNG { return &RNG{state: seed} }
func (r *RNG) Uint64() uint64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}
func (r *RNG) Float32() float32 { return float32(r.Uint64()>>40) * (1.0 / (1 << 24)) }
func (r *RNG) Intn(n int) int {
	if n <= 0 {
		panic("umap: invalid RNG bound")
	}
	bound := uint64(n)
	limit := ^uint64(0) - (^uint64(0) % bound)
	for {
		x := r.Uint64()
		if x < limit {
			return int(x % bound)
		}
	}
}

// intnBounded is Intn with its rejection bound precomputed for a hot loop.
func (r *RNG) intnBounded(bound, limit uint64) int {
	for {
		x := r.Uint64()
		if x < limit {
			return int(x % bound)
		}
	}
}
func (r *RNG) Shuffle(n int, swap func(i, j int)) {
	for i := n - 1; i > 0; i-- {
		swap(i, r.Intn(i+1))
	}
}
func DeriveSeed(seed uint64, stage string) uint64 {
	h := uint64(1469598103934665603)
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], seed)
	for _, v := range append(b[:], stage...) {
		h ^= uint64(v)
		h *= 1099511628211
	}
	return h
}
