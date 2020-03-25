package rchannel

import (
	"errors"
	"net"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

type CollectiveEndpoint struct {
	waitQ *BufferPool
	recvQ *BufferPool
}

func NewCollectiveEndpoint() *CollectiveEndpoint {
	return &CollectiveEndpoint{
		waitQ: newBufferPool(1),
		recvQ: newBufferPool(1),
	}
}

// Handle implements ConnHandler.Handle interface
func (e *CollectiveEndpoint) Handle(conn net.Conn, remote plan.NetAddr, t connection.ConnType) error {
	if t != connection.ConnCollective {
		return connection.ErrInvalidConnectionType
	}
	_, err := Stream(conn, remote, e.accept, e.handle)
	return err
}

func (e *CollectiveEndpoint) Recv(a plan.Addr) connection.Message {
	m := <-e.recvQ.require(a)
	return *m
}

var errRegisteredBufferNotUsed = errors.New("registered buffer not used")

func (e *CollectiveEndpoint) RecvInto(a plan.Addr, m connection.Message) error {
	e.waitQ.require(a) <- &m
	pm := <-e.recvQ.require(a)
	if !m.Same(pm) {
		return errRegisteredBufferNotUsed
	}
	return nil
}

func (e *CollectiveEndpoint) accept(conn net.Conn, remote plan.NetAddr) (string, *connection.Message, error) {
	var mh connection.MessageHeader
	if err := mh.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	name := string(mh.Name)
	if mh.HasFlag(connection.WaitRecvBuf) {
		m := <-e.waitQ.require(remote.WithName(name))
		if err := m.ReadInto(conn); err != nil {
			return "", nil, err
		}
		return name, m, nil
	}
	var m connection.Message
	if err := m.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	return name, &m, nil
}

func (e *CollectiveEndpoint) handle(name string, msg *connection.Message, conn net.Conn, remote plan.NetAddr) {
	e.recvQ.require(remote.WithName(name)) <- msg
}
