package utils

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"path"
	"path/filepath"
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

func LogNCCLEnv() {
	LogEnvWithPrefix(`NCCL_`, `nccl-env`)
}

func LogKungfuEnv() {
	LogEnvWithPrefix(`KUNGFU_`, `kf-env`)
}

func LogNICInfo() error {
	ifaces, err := net.Interfaces()
	if err != nil {
		return err
	}
	for i, nic := range ifaces {
		addrs, err := nic.Addrs()
		if err != nil {
			return err
		}
		var as []string
		for _, a := range addrs {
			as = append(as, a.String())
		}
		fmt.Printf("[nic] [%d] %s :: %s\n", i, nic.Name, strings.Join(as, ", "))
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

func ListNvidiaGPUNames() []string {
	const prefix = `/dev/`
	files, err := filepath.Glob(prefix + `nvidia*`)
	if err != nil {
		return nil
	}
	var names []string
	for _, file := range files {
		name := strings.TrimPrefix(file, prefix)
		var x int
		n, err := fmt.Sscanf(name, "nvidia%d", &x)
		if n == 1 && err == nil && fmt.Sprintf("nvidia%d", x) == name {
			names = append(names, name)
		}
	}
	sort.Strings(names) // FIXME: use numeric sort
	return names
}

func pluralize(n int, singular, plural string) string {
	if n > 1 {
		return plural
	}
	return singular
}

func Pluralize(n int, singular, plural string) string {
	return fmt.Sprintf("%d %s", n, pluralize(n, singular, plural))
}

func ProgName() string {
	if len(os.Args) > 0 {
		return path.Base(os.Args[0])
	}
	return ""
}

// Poll waits for a condition to become true
func Poll(ctx context.Context, cond func() bool) (int, bool) {
	for i := 0; ; i++ {
		if ok := cond(); ok {
			return i, true
		}
		select {
		case <-ctx.Done():
			return i + 1, false
		default:
			// user should sleep in cond
		}
	}
}

func ReadJSON(r io.Reader, i interface{}) error {
	d := json.NewDecoder(r)
	return d.Decode(&i)
}
