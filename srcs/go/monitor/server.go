package monitor

import (
	"net"
	"net/http"
	"strconv"

	"github.com/lsds/KungFu/srcs/go/log"
)

var (
	monitoringServer *http.Server
)

func StartServer(port int) {
	addr := net.JoinHostPort("0.0.0.0", strconv.Itoa(int(port)))
	monitoringServer = &http.Server{
		Handler: netMetrics,
		Addr:    addr,
	}
	go func() {
		if err := monitoringServer.ListenAndServe(); err != nil {
			if err != http.ErrServerClosed {
				log.Warnf("failed to start monitoring server: %v", err)
			}
		}
	}()
}

func StopServer() {
	monitoringServer.Close()
}
