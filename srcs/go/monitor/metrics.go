package monitor

import (
	"io"
	"net/http"
	"time"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
)

type NetMetrics struct {
	period time.Duration

	sent     *accumulator
	recv     *accumulator
	sentRete *rate
	recvRete *rate
}

var netMetrics *NetMetrics

func init() {
	netMetrics = newNetMetrics(kc.MonitoringPeriod)
}

func GetNetMetrics() *NetMetrics {
	return netMetrics
}

func newNetMetrics(p time.Duration) *NetMetrics {
	sent := newAccumulator("send")
	recv := newAccumulator("recv")
	m := &NetMetrics{
		period:   p,
		sent:     sent,
		recv:     recv,
		sentRete: newRate(sent, `_rate`),
		recvRete: newRate(recv, `_rate`),
	}
	if p > 0 {
		go m.start(p)
	}
	return m
}

func (m *NetMetrics) start(p time.Duration) {
	for range time.Tick(p) {
		m.sentRete.update()
		m.recvRete.update()
	}
}

func (m *NetMetrics) Sent(n int64) {
	m.sent.Add(n)
}

func (m *NetMetrics) Recv(n int64) {
	m.recv.Add(n)
}

func (m *NetMetrics) WriteTo(w io.Writer) {
	m.sent.WriteTo(w)
	m.recv.WriteTo(w)
	m.sentRete.WriteTo(w)
	m.recvRete.WriteTo(w)
}

func (m *NetMetrics) Handler() http.Handler {
	return &server{
		metrics: m,
	}
}
