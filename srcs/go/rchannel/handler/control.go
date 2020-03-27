package handler

import (
	"os"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

type ControlHandler struct {
}

func (h *ControlHandler) Handle(conn connection.Connection) (int, error) {
	return connection.Stream(conn, connection.Accept, h.handleControl)
}

func (h *ControlHandler) handleControl(name string, msg *connection.Message, _conn connection.Connection) {
	if name == "exit" {
		log.Errorf("exit control message received.")
		os.Exit(0)
	}
	log.Errorf("unexpected control message: %q", name)
}
