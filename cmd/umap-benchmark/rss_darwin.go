//go:build darwin

package main

import "syscall"

func peakRSS() (uint64, bool) {
	var usage syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &usage) != nil || usage.Maxrss <= 0 {
		return 0, false
	}
	return uint64(usage.Maxrss), true
}
