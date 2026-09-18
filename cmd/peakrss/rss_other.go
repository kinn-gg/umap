//go:build !linux && !darwin

package main

import "os"

func hardwareCPU() string { return os.Getenv("PROCESSOR_IDENTIFIER") }

func peakRSS(_ *os.ProcessState) (uint64, bool) { return 0, false }
