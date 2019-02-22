package monitor

import (
	"io"
	"net/http"
	"time"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type NetMetrics struct {
	egress  *rateAccumulator
	ingress *rateAccumulator
}

var netMetrics *NetMetrics

func init() {
	netMetrics = newNetMetrics(kc.MonitoringPeriod)
}

func GetNetMetrics() *NetMetrics {
	return netMetrics
}

func newNetMetrics(p time.Duration) *NetMetrics {
	m := &NetMetrics{
		egress:  newRateAccumulator("Egress rate in bytes"),
		ingress: newRateAccumulator("Ingress rate in bytes"),
	}
	if p > 0 {
		go m.start(p)
	}
	return m
}

func (m *NetMetrics) start(p time.Duration) {
	for range time.Tick(p) {
		m.egress.r.update()
		m.ingress.r.update()
	}
}

func (m *NetMetrics) Egress(n int64, a plan.Addr) {
	// TODO: add labels from Addr
	m.egress.a.Add(n)
}

func (m *NetMetrics) Ingress(n int64, a plan.Addr) {
	// TODO: add labels from Addr
	m.ingress.a.Add(n)
}

func (m *NetMetrics) WriteTo(w io.Writer) {
	m.egress.WriteTo(w)
	m.ingress.WriteTo(w)
}

func (m *NetMetrics) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	m.WriteTo(w)
}
