package utils

import (
	"fmt"
	"log"
	"net"
	"os"
	"sort"
	"strings"
	"time"
)

func LogArgs() {
	for i, a := range os.Args {
		fmt.Printf("[arg] [%d]=%s\n", i, a)
	}
}

func LogEnvWithPrefix(prefix string, logPrefix string) {
	for _, kv := range os.Environ() {
		if strings.HasPrefix(kv, prefix) {
			fmt.Printf("[%s]: %s\n", logPrefix, kv)
		}
	}
}

func LogCudaEnv() {
	LogEnvWithPrefix(`CUDA_`, `cuda-env`)
}

func LogKungfuEnv() {
	LogEnvWithPrefix(`KUNGFU_`, `kf-env`)
}

func LogNICInfo() error {
	ifaces, err := net.Interfaces()
	if err != nil {
		return err
	}
	for _, i := range ifaces {
		addrs, err := i.Addrs()
		if err != nil {
			return err
		}
		for _, a := range addrs {
			fmt.Printf("[nic] %s :: %s\n", i.Name, a)
		}
	}
	return nil
}

func LogAllEnvs() {
	envs := os.Environ()
	sort.Strings(envs)
	for _, e := range envs {
		fmt.Printf("[env] %s\n", e)
	}
}

func ExitErr(err error) {
	log.Printf("exit on error: %v", err)
	os.Exit(1)
}

func Measure(f func() error) (time.Duration, error) {
	t0 := time.Now()
	err := f()
	d := time.Since(t0)
	return d, err
}

func Rate(n int64, d time.Duration) float64 {
	return float64(n) / (float64(d) / float64(time.Second))
}

func ShowRate(r float64) string {
	const Ki = 1 << 10
	const Mi = 1 << 20
	const Gi = 1 << 30
	switch {
	case r > Gi:
		return fmt.Sprintf("%.2f GiB/s", r/float64(Gi))
	case r > Mi:
		return fmt.Sprintf("%.2f MiB/s", r/float64(Mi))
	case r > Ki:
		return fmt.Sprintf("%.2f KiB/s", r/float64(Ki))
	default:
		return fmt.Sprintf("%.2f B/s", r)
	}
}
