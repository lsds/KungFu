package rchannel

import (
	"io"
	"net"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type ConnHandler interface {
	Handle(conn net.Conn, remote plan.NetAddr, t ConnType) error
}

type Endpoint interface {
	Self() plan.PeerID
	ConnHandler
}

type acceptFunc func(conn net.Conn, remote plan.NetAddr) (string, *Message, error)

type msgHandleFunc func(name string, msg *Message, conn net.Conn, remote plan.NetAddr)

func stream(conn net.Conn, remote plan.NetAddr, accept acceptFunc, handle msgHandleFunc) (int, error) {
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
