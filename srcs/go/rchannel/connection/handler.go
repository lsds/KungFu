package connection

import (
	"io"

	"github.com/lsds/KungFu/srcs/go/log"
)

type Handler interface {
	Handle(conn Connection) (int, error)
}

type acceptFunc func(conn Connection) (string, *Message, error)

type MsgHandleFunc func(name string, msg *Message, conn Connection)

// Accept accepts one message from connection
func Accept(conn Connection) (string, *Message, error) {
	var mh MessageHeader
	if err := mh.ReadFrom(conn.Conn()); err != nil {
		return "", nil, err
	}
	var msg Message // FIXME: don't use buf
	if err := msg.ReadFrom(conn.Conn()); err != nil {
		return "", nil, err
	}
	return string(mh.Name), &msg, nil
}

func Stream(conn Connection, accept acceptFunc, handle MsgHandleFunc) (int, error) {
	for i := 0; ; i++ {
		name, msg, err := accept(conn)
		if err != nil {
			if err == io.EOF {
				return i, nil
			}
			log.Warnf("accept message error: %v", err)
			return i, err
		}
		handle(name, msg, conn)
	}
}
