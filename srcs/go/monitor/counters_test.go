package monitor

import (
	"bytes"
	"testing"

	"github.com/lsds/KungFu/srcs/go/plan"
)

func Test_rateAccumulator(t *testing.T) {
	var b bytes.Buffer
	var a plan.Addr
	nm := newNetMetrics(0)
	nm.Egress(3, a)
	nm.Ingress(2, a)
	nm.WriteTo(&b)
	const want = `egress_total_bytes 3
egress_rate_bytes_per_sec 0.000000
ingress_total_bytes 2
ingress_rate_bytes_per_sec 0.000000
`
	if got := b.String(); got != want {
		t.Errorf("want %q, got %q", want, got)
	}
}
