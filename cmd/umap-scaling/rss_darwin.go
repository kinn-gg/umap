//go:build darwin

package main

import (
	"os/exec"
	"strings"
	"syscall"
)

func peakRSS() (uint64, bool) {
	var usage syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &usage) != nil || usage.Maxrss <= 0 {
		return 0, false
	}
	return uint64(usage.Maxrss), true
}

func hardwareCPU() string {
	b, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}
