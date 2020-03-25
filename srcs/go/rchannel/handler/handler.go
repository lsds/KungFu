package handler

import (
	"io"
	"net"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

type ConnHandler interface {
	Handle(conn net.Conn, remote plan.NetAddr, t connection.ConnType) error
}

type Endpoint interface {
	Self() plan.PeerID
	ConnHandler
}

type acceptFunc func(conn net.Conn, remote plan.NetAddr) (string, *connection.Message, error)

type MsgHandleFunc func(name string, msg *connection.Message, conn net.Conn, remote plan.NetAddr)

// Accept accepts one message from connection
func Accept(conn net.Conn, _remote plan.NetAddr) (string, *connection.Message, error) {
	var mh connection.MessageHeader
	if err := mh.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	var msg connection.Message // FIXME: don't use buf
	if err := msg.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	return string(mh.Name), &msg, nil
}

func Stream(conn net.Conn, remote plan.NetAddr, accept acceptFunc, handle MsgHandleFunc) (int, error) {
	for i := 0; ; i++ {
		name, msg, err := accept(conn, remote)
		if err != nil {
			if err == io.EOF {
				return i, nil
			}
			log.Warnf("accept message error: %v", err)
			return i, err
		}
		handle(name, msg, conn, remote)
	}
}
