package rchannel

import (
	"fmt"
	"net"
	"os"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type controlHandler struct {
}

func (h *controlHandler) Handle(conn net.Conn, remote plan.NetAddr, t ConnType) error {
	switch t {
	case ConnControl:
		if n, err := Stream(conn, remote, Accept, h.handleControl); err != nil {
			return fmt.Errorf("stream error after handled %d messages: %v", n, err)
		}
		return nil
	default:
		return ErrInvalidConnectionType
	}
}

func (h *controlHandler) handleControl(name string, msg *Message, _conn net.Conn, remote plan.NetAddr) {
	if name == "exit" {
		log.Errorf("exit control message received.")
		os.Exit(0)
	}
	log.Errorf("unexpected control message: %q", name)
}
