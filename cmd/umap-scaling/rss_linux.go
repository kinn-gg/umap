//go:build linux

package main

import (
	"os"
	"strings"
	"syscall"
)

func peakRSS() (uint64, bool) {
	var usage syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &usage) != nil || usage.Maxrss <= 0 {
		return 0, false
	}
	return uint64(usage.Maxrss) * 1024, true
}

func hardwareCPU() string {
	b, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(b), "\n") {
		if key, value, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(key) == "model name" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
