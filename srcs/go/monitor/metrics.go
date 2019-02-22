package monitor

import (
	"io"
	"net/http"
	"time"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type NetMetrics struct {
	sent *rateAccumulator
	recv *rateAccumulator
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
		sent: newRateAccumulator("send"),
		recv: newRateAccumulator("recv"),
	}
	if p > 0 {
		go m.start(p)
	}
	return m
}

func (m *NetMetrics) start(p time.Duration) {
	for range time.Tick(p) {
		m.sent.r.update()
		m.recv.r.update()
	}
}

func (m *NetMetrics) Sent(n int64, a plan.Addr) {
	// TODO: add labels from Addr
	m.sent.a.Add(n)
}

func (m *NetMetrics) Recv(n int64, a plan.Addr) {
	// TODO: add labels from Addr
	m.recv.a.Add(n)
}

func (m *NetMetrics) WriteTo(w io.Writer) {
	m.sent.WriteTo(w)
	m.recv.WriteTo(w)
}

func (m *NetMetrics) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	m.WriteTo(w)
}
