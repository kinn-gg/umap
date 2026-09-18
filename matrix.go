package umap

import (
	"math"
	"slices"
)

type Matrix interface {
	Shape() (rows, columns int)
	matrix()
}

type Dense struct {
	data                  []float32
	rows, columns, stride int
}

func NewDense(data []float32, rows, columns int) (Dense, error) {
	return NewDenseStride(data, rows, columns, columns)
}

func NewDenseStride(data []float32, rows, columns, stride int) (Dense, error) {
	if rows < 0 || columns <= 0 || stride < columns {
		return Dense{}, shapef("invalid dense shape %dx%d with stride %d", rows, columns, stride)
	}
	need := 0
	if rows > 0 {
		if stride > (int(^uint(0)>>1)-columns)/max(1, rows-1) {
			return Dense{}, shapef("dense dimensions overflow")
		}
		need = (rows-1)*stride + columns
	}
	if len(data) < need {
		return Dense{}, shapef("dense data has %d values, need %d", len(data), need)
	}
	for i := 0; i < rows; i++ {
		for _, v := range data[i*stride : i*stride+columns] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return Dense{}, numericf("dense matrix contains non-finite value")
			}
		}
	}
	return Dense{data: data[:need], rows: rows, columns: columns, stride: stride}, nil
}
func (Dense) matrix()             {}
func (d Dense) Shape() (int, int) { return d.rows, d.columns }
func (d Dense) At(r, c int) float32 {
	if r < 0 || r >= d.rows || c < 0 || c >= d.columns {
		panic("umap: dense index out of range")
	}
	return d.data[r*d.stride+c]
}
func (d Dense) Row(r int) []float32 {
	if r < 0 || r >= d.rows {
		panic("umap: dense row out of range")
	}
	return d.data[r*d.stride : r*d.stride+d.columns]
}

type CSR struct {
	values            []float32
	columns           []uint32
	offsets           []uint64
	rows, columnCount int
}

func NewCSR(values []float32, columns []uint32, rowOffsets []uint64, rows, columnCount int) (CSR, error) {
	if rows < 0 || columnCount <= 0 || len(rowOffsets) != rows+1 || len(values) != len(columns) {
		return CSR{}, shapef("invalid CSR dimensions")
	}
	if len(rowOffsets) == 0 || rowOffsets[0] != 0 || rowOffsets[len(rowOffsets)-1] != uint64(len(values)) {
		return CSR{}, validationf("invalid CSR row offsets")
	}
	for r := 0; r < rows; r++ {
		if rowOffsets[r] > rowOffsets[r+1] || rowOffsets[r+1] > uint64(len(values)) {
			return CSR{}, validationf("invalid CSR row offsets")
		}
		var prev uint32
		for p := rowOffsets[r]; p < rowOffsets[r+1]; p++ {
			c := columns[p]
			if uint64(c) >= uint64(columnCount) || (p > rowOffsets[r] && c <= prev) {
				return CSR{}, validationf("CSR columns must be sorted, unique, and in range")
			}
			prev = c
			v := values[p]
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return CSR{}, numericf("CSR contains non-finite value")
			}
		}
	}
	return CSR{values, columns, rowOffsets, rows, columnCount}, nil
}
func (CSR) matrix()             {}
func (c CSR) Shape() (int, int) { return c.rows, c.columnCount }
func (c CSR) Row(r int) ([]uint32, []float32) {
	if r < 0 || r >= c.rows {
		panic("umap: CSR row out of range")
	}
	a, b := c.offsets[r], c.offsets[r+1]
	return c.columns[a:b], c.values[a:b]
}
func (c CSR) At(r, col int) float32 {
	cs, vs := c.Row(r)
	i, ok := slices.BinarySearch(cs, uint32(col))
	if ok {
		return vs[i]
	}
	return 0
}

func denseRow(m Matrix, r int, scratch []float32) []float32 {
	switch x := m.(type) {
	case Dense:
		return x.Row(r)
	case CSR:
		clear(scratch)
		cs, vs := x.Row(r)
		for i, c := range cs {
			scratch[c] = vs[i]
		}
		return scratch
	default:
		panic("unreachable")
	}
}
