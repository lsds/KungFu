package monitor

import (
	"bytes"
	"testing"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/plan"
)

func Test_rateAccumulator(t *testing.T) {
	kc.EnableMonitoring = true // FIXME: don't modify global variable
	var b bytes.Buffer
	var a plan.NetAddr
	nm := newMonitor(0)
	nm.Egress(3, a)
	nm.Ingress(2, a)
	nm.writeTo(&b)

	// FIXME: add assert
	// 	const want = `egress_total_bytes 3
	// egress_rate_bytes_per_sec 0.000000
	// ingress_total_bytes 2
	// ingress_rate_bytes_per_sec 0.000000
	// `
	// 	if got := b.String(); got != want {
	// 		t.Errorf("want %q, got %q", want, got)
	// 	}
}
