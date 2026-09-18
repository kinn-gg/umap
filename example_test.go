package umap_test

import (
	"context"
	"fmt"
	"log"

	"github.com/kinn-gg/umap"
)

func Example() {
	x, err := umap.NewDense([]float32{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
	if err != nil {
		log.Fatal(err)
	}
	seed := uint64(7)
	cfg := umap.DefaultConfig()
	cfg.Neighbors, cfg.Epochs, cfg.Seed = 3, 10, &seed
	reducer, err := umap.New(cfg)
	if err != nil {
		log.Fatal(err)
	}
	model, embedding, err := reducer.FitTransform(context.Background(), x, umap.NoTarget())
	if err != nil {
		log.Fatal(err)
	}
	projected, err := model.Transform(context.Background(), x)
	if err != nil {
		log.Fatal(err)
	}
	r, c := embedding.Shape()
	pr, pc := projected.Shape()
	fmt.Println(r, c, pr, pc)
	// Output: 4 2 4 2
}

func ExampleModel_MarshalBinary() {
	x, _ := umap.NewDense([]float32{0, 0, 1, 1, 2, 2}, 3, 2)
	cfg := umap.DefaultConfig()
	cfg.Neighbors, cfg.Epochs = 2, 2
	reducer, _ := umap.New(cfg)
	model, _ := reducer.Fit(context.Background(), x, umap.NoTarget())
	encoded, err := model.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	restored, err := umap.UnmarshalModel(encoded)
	if err != nil {
		log.Fatal(err)
	}
	r, c := restored.Embedding().Shape()
	fmt.Println(r, c)
	// Output: 3 2
}
