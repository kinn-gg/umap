package main

import "testing"

func TestWorkloadsAreReproducible(t *testing.T) {
	for _, kind := range []string{"dense", "sparse"} {
		a, err := makeWorkload(kind, 100, 2, 3)
		if err != nil {
			t.Fatal(err)
		}
		b, err := makeWorkload(kind, 100, 2, 3)
		if err != nil {
			t.Fatal(err)
		}
		if a.sum != b.sum {
			t.Errorf("%s checksums differ: %s != %s", kind, a.sum, b.sum)
		}
		if a.params.Backend != "nn_descent" || a.params.Workers != 2 {
			t.Errorf("%s effective search settings not recorded: %+v", kind, a.params)
		}
	}
}

func TestRegressionBudgetUsesRepeatedSampleDispersion(t *testing.T) {
	got := deriveBudget([]int64{100, 110, 120})
	if got.MedianNS != 110 || got.MADNS != 10 || got.LimitNS != 170 {
		t.Fatalf("budget = %+v", got)
	}
}
