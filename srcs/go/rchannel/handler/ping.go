package handler

import (
	"net"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

type PingHandler struct {
}

func (h *PingHandler) Handle(conn net.Conn, remote plan.NetAddr, t connection.ConnType) error {
	var mh connection.MessageHeader
	if err := mh.ReadFrom(conn); err != nil {
		return err
	}
	var empty connection.Message
	if err := empty.ReadFrom(conn); err != nil {
		return err
	}
	if err := mh.WriteTo(conn); err != nil {
		return err
	}
	return empty.WriteTo(conn)
}
