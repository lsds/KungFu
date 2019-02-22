package monitor

import (
	"net"
	"net/http"
	"strconv"
)

type server struct {
	metrics *NetMetrics
}

func (s *server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	s.metrics.WriteTo(w)
}

func ListenAndServe(port int) {
	addr := net.JoinHostPort("0.0.0.0", strconv.Itoa(int(port)))
	http.ListenAndServe(addr, netMetrics.Handler())
}
