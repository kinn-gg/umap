//go:build !linux && !darwin

package main

func peakRSS() (uint64, bool) { return 0, false }
