// Command peakrss runs a process and emits a machine-readable resource record.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"time"
)

type record struct {
	SchemaVersion int      `json:"schema_version"`
	Command       []string `json:"command"`
	Platform      string   `json:"platform"`
	Architecture  string   `json:"architecture"`
	CPU           string   `json:"cpu,omitempty"`
	GoVersion     string   `json:"go_version"`
	WallNS        int64    `json:"wall_ns"`
	PeakRSSBytes  *uint64  `json:"peak_rss_bytes"`
	ExitCode      int      `json:"exit_code"`
}

func main() {
	output := flag.String("output", "", "write JSON record to this file (default stdout)")
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: peakrss [--output path] -- command [args...]")
		os.Exit(2)
	}
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	start := time.Now()
	err := cmd.Run()
	elapsed := time.Since(start)
	exitCode := 0
	if err != nil {
		if e, ok := err.(*exec.ExitError); ok {
			exitCode = e.ExitCode()
		} else {
			fmt.Fprintln(os.Stderr, err)
			exitCode = -1
		}
	}
	r := record{SchemaVersion: 1, Command: args, Platform: runtime.GOOS, Architecture: runtime.GOARCH, CPU: hardwareCPU(), GoVersion: runtime.Version(), WallNS: elapsed.Nanoseconds(), ExitCode: exitCode}
	if value, ok := peakRSS(cmd.ProcessState); ok {
		r.PeakRSSBytes = &value
	}
	var w *os.File
	if *output == "" {
		w = os.Stdout
	} else {
		var openErr error
		w, openErr = os.Create(*output)
		if openErr != nil {
			fmt.Fprintln(os.Stderr, openErr)
			os.Exit(2)
		}
		defer w.Close()
	}
	if encodeErr := json.NewEncoder(w).Encode(r); encodeErr != nil {
		fmt.Fprintln(os.Stderr, encodeErr)
		os.Exit(2)
	}
	if exitCode != 0 {
		os.Exit(exitCode)
	}
}
