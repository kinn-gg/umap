//go:build darwin

package main

import (
	"os"
	"os/exec"
	"strings"
	"syscall"
)

func hardwareCPU() string {
	out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func peakRSS(state *os.ProcessState) (uint64, bool) {
	r, ok := state.SysUsage().(*syscall.Rusage)
	if !ok || r.Maxrss < 0 {
		return 0, false
	}
	return uint64(r.Maxrss), true
}
