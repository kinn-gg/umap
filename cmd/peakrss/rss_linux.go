//go:build linux

package main

import (
	"bufio"
	"os"
	"strings"
	"syscall"
)

func hardwareCPU() string {
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		if strings.HasPrefix(line, "model name") {
			if _, value, ok := strings.Cut(line, ":"); ok {
				return strings.TrimSpace(value)
			}
		}
	}
	return ""
}

func peakRSS(state *os.ProcessState) (uint64, bool) {
	r, ok := state.SysUsage().(*syscall.Rusage)
	if !ok || r.Maxrss < 0 {
		return 0, false
	}
	return uint64(r.Maxrss) * 1024, true
}
