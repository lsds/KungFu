package testutils

import (
	"fmt"
	"time"
)

const (
	Ki = 1 << 10
	Mi = 1 << 20
	Gi = 1 << 30
)

func ShowSize(n int64) string {
	switch {
	case n < Ki:
		return fmt.Sprintf("%dB", n)
	case n < Mi:
		return fmt.Sprintf("%dKiB", n/Ki)
	case n < Gi:
		return fmt.Sprintf("%dMiB", n/Mi)
	default:
		return fmt.Sprintf("%dGiB", n/Gi)
	}
}

func ShowRate(n int64, d time.Duration) string {
	rate := float64(n) / (float64(d) / float64(time.Second))
	switch {
	case rate < float64(Ki):
		return fmt.Sprintf("%.2fB/s", rate)
	case rate < float64(Mi):
		return fmt.Sprintf("%.2fKiB/s", rate/float64(Ki))
	case rate < float64(Gi):
		return fmt.Sprintf("%.2fMiB/s", rate/float64(Mi))
	default:
		return fmt.Sprintf("%.2fGiB/s", rate/float64(Gi))
	}
}
