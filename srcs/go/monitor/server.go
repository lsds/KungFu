package monitor

import (
	"net"
	"net/http"
	"strconv"

	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	monitoringServer *http.Server
)

func StartServer(port int) {
	addr := net.JoinHostPort("0.0.0.0", strconv.Itoa(int(port)))
	monitoringServer = &http.Server{
		Handler: defaultMonitor,
		Addr:    addr,
	}
	go func() {
		if err := monitoringServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			utils.ExitErr(err)
		}
	}()
}

func StopServer() {
	monitoringServer.Close()
}
