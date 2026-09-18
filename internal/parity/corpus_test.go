package parity

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCheckedInFastCorpus(t *testing.T) {
	dir := filepath.Join("..", "..", "parity", "fixtures")
	full := false
	if configured := os.Getenv("PARITY_CORPUS"); configured != "" {
		dir = configured
		full = true
	}
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		SchemaVersion   int    `json:"schema_version"`
		ReferenceCommit string `json:"reference_commit"`
		Fixtures        []struct {
			Name        string `json:"name"`
			Suite       string `json:"suite"`
			File        string `json:"file"`
			InputSHA256 string `json:"input_sha256"`
		} `json:"fixtures"`
	}
	if err := json.Unmarshal(b, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.SchemaVersion != SchemaVersion || len(manifest.Fixtures) == 0 {
		t.Fatalf("invalid manifest: %+v", manifest)
	}
	for _, f := range manifest.Fixtures {
		t.Run(f.Name, func(t *testing.T) {
			if !full && f.Suite != "fast" {
				t.Fatalf("checked-in fixture %q is not in fast suite", f.Name)
			}
			a, err := Load(filepath.Join(dir, f.File))
			if err != nil {
				t.Fatal(err)
			}
			if a.Name != f.Name || a.Provenance.InputSHA256 != f.InputSHA256 || a.Provenance.ReferenceCommit != manifest.ReferenceCommit {
				t.Fatal("artifact provenance does not match manifest")
			}
		})
	}
	if full && len(manifest.Fixtures) != 6 {
		t.Fatalf("full corpus has %d fixtures, want 6", len(manifest.Fixtures))
	}
}
