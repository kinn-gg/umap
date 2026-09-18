// Command umap-parity compares a candidate artifact with a Python reference.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/kinn-gg/umap/internal/parity"
)

func main() {
	refPath := flag.String("reference", "", "reference artifact JSON")
	gotPath := flag.String("candidate", "", "candidate artifact JSON")
	flag.Parse()
	if *refPath == "" || *gotPath == "" {
		flag.Usage()
		os.Exit(2)
	}
	ref, err := parity.Load(*refPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	got, err := parity.Load(*gotPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	r := parity.Compare(ref, got, parity.DefaultThresholds)
	_ = json.NewEncoder(os.Stdout).Encode(r)
	if !r.Passed {
		os.Exit(1)
	}
}
