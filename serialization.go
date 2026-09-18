package umap

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"math"
)

const (
	modelMagic   = "UMAPGO\x00\x01"
	maxModelSize = 256 << 20
)

type serialConfig struct {
	Neighbors, Components, Epochs, NegativeSampleRate int
	Metric                                            Metric
	LearningRate, MinDist, Spread, SetOpMixRatio      float32
	LocalConnectivity, RepulsionStrength, A, B        float32
	Init                                              Init
	TransformSeed                                     uint64
	Target                                            TargetConfig
}

type serialMatrix struct {
	Kind           string
	Data, Values   []float32
	Columns        []uint32
	Offsets        []uint64
	Rows, Features int
}

type serialModel struct {
	Version   int
	Config    serialConfig
	Seed      uint64
	Training  serialMatrix
	Embedding []float32
	Graph     Graph
}

func (m *Model) MarshalBinary() ([]byte, error) {
	if m == nil || m.embedding == nil || m.training == nil {
		return nil, validationf("model is not fitted")
	}
	sm := serialModel{Version: 1, Config: configForSerialization(m.config), Seed: m.seed, Embedding: m.embedding.data, Graph: m.graph}
	switch x := m.training.(type) {
	case Dense:
		sm.Training = serialMatrix{Kind: "dense", Data: x.data, Rows: x.rows, Features: x.columns}
	case CSR:
		sm.Training = serialMatrix{Kind: "csr", Values: x.values, Columns: x.columns, Offsets: x.offsets, Rows: x.rows, Features: x.columnCount}
	default:
		return nil, unsupportedf("unsupported training matrix type")
	}
	payload, err := json.Marshal(sm)
	if err != nil {
		return nil, validationf("encode model: %v", err)
	}
	if len(payload) > maxModelSize {
		return nil, validationf("model exceeds maximum encoded size")
	}
	sum := sha256.Sum256(payload)
	out := make([]byte, len(modelMagic)+8+len(sum)+len(payload))
	copy(out, modelMagic)
	binary.LittleEndian.PutUint64(out[len(modelMagic):], uint64(len(payload)))
	copy(out[len(modelMagic)+8:], sum[:])
	copy(out[len(modelMagic)+8+len(sum):], payload)
	return out, nil
}

func UnmarshalModel(data []byte) (*Model, error) {
	header := len(modelMagic) + 8 + sha256.Size
	if len(data) < header || string(data[:len(modelMagic)]) != modelMagic {
		return nil, validationf("invalid model header")
	}
	n := binary.LittleEndian.Uint64(data[len(modelMagic):])
	if n > maxModelSize || n != uint64(len(data)-header) {
		return nil, validationf("invalid or oversized model payload")
	}
	payload := data[header:]
	want := data[len(modelMagic)+8 : header]
	got := sha256.Sum256(payload)
	if string(want) != string(got[:]) {
		return nil, validationf("model checksum mismatch")
	}
	var sm serialModel
	if err := json.Unmarshal(payload, &sm); err != nil {
		return nil, validationf("decode model: %v", err)
	}
	if sm.Version != 1 || sm.Training.Rows < 2 || sm.Training.Features < 1 || sm.Config.Components < 1 || sm.Graph.Vertices != sm.Training.Rows || sm.Training.Rows > int(^uint(0)>>1)/sm.Config.Components || len(sm.Embedding) != sm.Training.Rows*sm.Config.Components {
		return nil, validationf("invalid model contents")
	}
	cfg := configFromSerialization(sm.Config)
	if _, err := New(cfg); err != nil {
		return nil, err
	}
	var training Matrix
	var err error
	switch sm.Training.Kind {
	case "dense":
		training, err = NewDense(sm.Training.Data, sm.Training.Rows, sm.Training.Features)
	case "csr":
		training, err = NewCSR(sm.Training.Values, sm.Training.Columns, sm.Training.Offsets, sm.Training.Rows, sm.Training.Features)
	default:
		err = validationf("unknown model matrix encoding")
	}
	if err != nil {
		return nil, err
	}
	for _, e := range sm.Graph.Edges {
		if e.Head < 0 || e.Head >= sm.Training.Rows || e.Tail < 0 || e.Tail >= sm.Training.Rows || e.Weight < 0 || math.IsNaN(float64(e.Weight)) || math.IsInf(float64(e.Weight), 0) {
			return nil, validationf("invalid serialized graph")
		}
	}
	for _, v := range sm.Embedding {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, numericf("serialized embedding contains non-finite value")
		}
	}
	emb := &Embedding{data: append([]float32(nil), sm.Embedding...), rows: sm.Training.Rows, components: cfg.Components}
	return &Model{embedding: emb, graph: sm.Graph, seed: sm.Seed, config: cfg, training: training}, nil
}

func configForSerialization(c Config) serialConfig {
	return serialConfig{c.Neighbors, c.Components, c.Epochs, c.NegativeSampleRate, c.Metric, c.LearningRate, c.MinDist, c.Spread, c.SetOpMixRatio, c.LocalConnectivity, c.RepulsionStrength, c.A, c.B, c.Init, c.TransformSeed, c.Target}
}

func configFromSerialization(s serialConfig) Config {
	c := DefaultConfig()
	c.Neighbors, c.Components, c.Epochs, c.NegativeSampleRate = s.Neighbors, s.Components, s.Epochs, s.NegativeSampleRate
	c.Metric, c.LearningRate, c.MinDist, c.Spread = s.Metric, s.LearningRate, s.MinDist, s.Spread
	c.SetOpMixRatio, c.LocalConnectivity, c.RepulsionStrength = s.SetOpMixRatio, s.LocalConnectivity, s.RepulsionStrength
	c.A, c.B, c.Init, c.TransformSeed, c.Target = s.A, s.B, s.Init, s.TransformSeed, s.Target
	seed := uint64(0)
	c.Seed = &seed
	return c
}
