package handler

import (
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

type PingHandler struct {
}

func (h *PingHandler) Handle(conn connection.Connection) (int, error) {
	name, msg, err := connection.Accept(conn)
	if err != nil {
		return 0, err
	}
	if err := conn.Send(name, *msg, connection.NoFlag); err != nil {
		return 1, err
	}
	return 1, nil
}
