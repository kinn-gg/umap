// Package parity compares staged UMAP pipeline artifacts with a pinned reference.
package parity

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
)

const SchemaVersion = 1

type Provenance struct {
	Generator       string            `json:"generator"`
	ReferenceCommit string            `json:"reference_commit"`
	Python          string            `json:"python"`
	Platform        string            `json:"platform"`
	Packages        map[string]string `json:"packages"`
	InputSHA256     string            `json:"input_sha256"`
	Seed            int64             `json:"seed"`
}

type Parameters struct {
	Neighbors         int     `json:"n_neighbors"`
	Components        int     `json:"n_components"`
	Metric            string  `json:"metric"`
	MinDist           float64 `json:"min_dist"`
	Spread            float64 `json:"spread"`
	LocalConnectivity float64 `json:"local_connectivity"`
	Epochs            int     `json:"n_epochs"`
}

type Matrix struct {
	Rows    int       `json:"rows"`
	Columns int       `json:"columns"`
	Data    []float64 `json:"data,omitempty"`
	Values  []float64 `json:"values,omitempty"`
	Indices []int     `json:"indices,omitempty"`
	Indptr  []int     `json:"indptr,omitempty"`
}

func (m Matrix) Dense() ([]float64, error) {
	if len(m.Data) != 0 {
		if len(m.Data) != m.Rows*m.Columns {
			return nil, fmt.Errorf("dense data has %d values, want %d", len(m.Data), m.Rows*m.Columns)
		}
		return m.Data, nil
	}
	if len(m.Indptr) != m.Rows+1 || len(m.Values) != len(m.Indices) {
		return nil, errors.New("invalid CSR matrix")
	}
	out := make([]float64, m.Rows*m.Columns)
	for r := 0; r < m.Rows; r++ {
		for p := m.Indptr[r]; p < m.Indptr[r+1]; p++ {
			if m.Indices[p] < 0 || m.Indices[p] >= m.Columns {
				return nil, errors.New("CSR column out of range")
			}
			out[r*m.Columns+m.Indices[p]] = m.Values[p]
		}
	}
	return out, nil
}

type KNN struct {
	Indices   []int     `json:"indices"`
	Distances []float64 `json:"distances"`
}
type SmoothKNN struct {
	Rho   []float64 `json:"rho"`
	Sigma []float64 `json:"sigma"`
}
type Edge struct {
	Head   int     `json:"head"`
	Tail   int     `json:"tail"`
	Weight float64 `json:"weight"`
}

type Artifact struct {
	SchemaVersion  int        `json:"schema_version"`
	Name           string     `json:"name"`
	Suite          string     `json:"suite"`
	Provenance     Provenance `json:"provenance"`
	Parameters     Parameters `json:"parameters"`
	Input          Matrix     `json:"input"`
	KNN            KNN        `json:"knn"`
	SmoothKNN      SmoothKNN  `json:"smooth_knn"`
	Graph          []Edge     `json:"fuzzy_graph"`
	Initialization []float64  `json:"initialization"`
	Embedding      []float64  `json:"embedding"`
}

func Load(path string) (*Artifact, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var a Artifact
	if err := json.Unmarshal(b, &a); err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("validate %s: %w", path, err)
	}
	return &a, nil
}

func (a *Artifact) Validate() error {
	if a.SchemaVersion != SchemaVersion {
		return fmt.Errorf("schema version %d, want %d", a.SchemaVersion, SchemaVersion)
	}
	if a.Input.Rows < 2 || a.Input.Columns < 1 {
		return errors.New("invalid input shape")
	}
	dense, err := a.Input.Dense()
	if err != nil {
		return err
	}
	if a.Provenance.InputSHA256 != "" {
		b := make([]byte, 4*len(dense))
		for i, v := range dense {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(v)))
		}
		sum := sha256.Sum256(b)
		if hex.EncodeToString(sum[:]) != a.Provenance.InputSHA256 {
			return errors.New("input checksum mismatch")
		}
	}
	k := a.Parameters.Neighbors
	if len(a.KNN.Indices) != a.Input.Rows*k || len(a.KNN.Distances) != a.Input.Rows*k {
		return errors.New("invalid kNN shape")
	}
	if len(a.SmoothKNN.Rho) != a.Input.Rows || len(a.SmoothKNN.Sigma) != a.Input.Rows {
		return errors.New("invalid smooth kNN shape")
	}
	n := a.Input.Rows * a.Parameters.Components
	if len(a.Initialization) != n || len(a.Embedding) != n {
		return errors.New("invalid embedding shape")
	}
	return nil
}
