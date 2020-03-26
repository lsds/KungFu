package handler

import (
	"io"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

type acceptFunc func(conn connection.Connection) (string, *connection.Message, error)

type MsgHandleFunc func(name string, msg *connection.Message, conn connection.Connection)

// Accept accepts one message from connection
func Accept(conn connection.Connection) (string, *connection.Message, error) {
	var mh connection.MessageHeader
	if err := mh.ReadFrom(conn.Conn()); err != nil {
		return "", nil, err
	}
	var msg connection.Message // FIXME: don't use buf
	if err := msg.ReadFrom(conn.Conn()); err != nil {
		return "", nil, err
	}
	return string(mh.Name), &msg, nil
}

func Stream(conn connection.Connection, accept acceptFunc, handle MsgHandleFunc) (int, error) {
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
