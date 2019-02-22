package monitor

import (
	"io"
	"net/http"
	"time"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type netMonitor interface {
	Egress(n int64, a plan.Addr)
	Ingress(n int64, a plan.Addr)
}

type Monitor interface {
	http.Handler

	netMonitor

	writeTo(w io.Writer) // For testing
}

var monitor Monitor

func init() {
	monitor = newMonitor(kc.MonitoringPeriod)
}

func GetMonitor() Monitor {
	return monitor
}

type noopMonitor struct {
}

func (m *noopMonitor) Egress(n int64, a plan.Addr) {}

func (m *noopMonitor) Ingress(n int64, a plan.Addr) {}

func (m *noopMonitor) ServeHTTP(w http.ResponseWriter, req *http.Request) {}

func (m *noopMonitor) writeTo(w io.Writer) {}

type netMetrics struct {
	egress  *rateAccumulator
	ingress *rateAccumulator
}

func newMonitor(p time.Duration) Monitor {
	if !kc.EnableMonitoring {
		return &noopMonitor{}
	}
	m := &netMetrics{
		egress:  newRateAccumulator("egress"),
		ingress: newRateAccumulator("ingress"),
	}
	if p > 0 {
		go m.start(p)
	}
	return m
}

func (m *netMetrics) start(p time.Duration) {
	for range time.Tick(p) {
		m.egress.r.update(p)
		m.ingress.r.update(p)
	}
}

func (m *netMetrics) Egress(n int64, a plan.Addr) {
	// TODO: add labels from Addr
	m.egress.a.Add(n)
}

func (m *netMetrics) Ingress(n int64, a plan.Addr) {
	// TODO: add labels from Addr
	m.ingress.a.Add(n)
}

func (m *netMetrics) WriteTo(w io.Writer) {
	m.egress.WriteTo(w)
	m.ingress.WriteTo(w)
}

func (m *netMetrics) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	m.WriteTo(w)
}

// For testing
func (m *netMetrics) writeTo(w io.Writer) {
	m.WriteTo(w)
}
