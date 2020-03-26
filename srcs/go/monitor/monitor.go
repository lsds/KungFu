package monitor

import (
	"io"
	"net/http"
	"time"

	kc "github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type netMonitor interface {
	Egress(n int64, a plan.NetAddr)
	Ingress(n int64, a plan.NetAddr)
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

func (m *noopMonitor) Egress(n int64, a plan.NetAddr) {}

func (m *noopMonitor) Ingress(n int64, a plan.NetAddr) {}

func (m *noopMonitor) ServeHTTP(w http.ResponseWriter, req *http.Request) {}

func (m *noopMonitor) writeTo(w io.Writer) {}

type netMetrics struct {
	egressCounters  *rateAccumulatorGroup
	ingressCounters *rateAccumulatorGroup
}

func newMonitor(p time.Duration) Monitor {
	if !kc.EnableMonitoring {
		return &noopMonitor{}
	}
	m := &netMetrics{
		egressCounters:  newRateAccumulatorGroup("egress"),
		ingressCounters: newRateAccumulatorGroup("ingress"),
	}
	if p > 0 {
		go m.start(p)
	}
	return m
}

func (m *netMetrics) start(p time.Duration) {
	for range time.Tick(p) {
		m.egressCounters.update(p)
		m.ingressCounters.update(p)
	}
}

func (m *netMetrics) Egress(n int64, a plan.NetAddr) {
	ra := m.egressCounters.getOrCreate(a)
	ra.a.Add(n)
}

func (m *netMetrics) Ingress(n int64, a plan.NetAddr) {
	ra := m.ingressCounters.getOrCreate(a)
	ra.a.Add(n)
}

func (m *netMetrics) WriteTo(w io.Writer) {
	m.egressCounters.WriteTo(w)
	m.ingressCounters.WriteTo(w)
}

func (m *netMetrics) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	m.WriteTo(w)
}

// For testing
func (m *netMetrics) writeTo(w io.Writer) {
	m.WriteTo(w)
}
