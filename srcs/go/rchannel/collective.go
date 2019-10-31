package rchannel

import (
	"errors"
	"net"

	"github.com/lsds/KungFu/srcs/go/plan"
)

type CollectiveEndpoint struct {
	waitQ *BufferPool
	recvQ *BufferPool
}

func NewCollectiveEndpoint() *CollectiveEndpoint {
	return &CollectiveEndpoint{
		waitQ: newBufferPool(),
		recvQ: newBufferPool(),
	}
}

// Handle implements ConnHandler.Handle interface
func (e *CollectiveEndpoint) Handle(conn net.Conn, remote plan.NetAddr, t ConnType) error {
	if t != ConnCollective {
		return ErrInvalidConnectionType
	}
	_, err := Stream(conn, remote, e.accept, e.handle)
	return err
}

func (e *CollectiveEndpoint) Recv(a plan.Addr) Message {
	m := <-e.recvQ.require(a)
	return *m
}

var errRegisteredBufferNotUsed = errors.New("registered buffer not used")

func (e *CollectiveEndpoint) RecvInto(a plan.Addr, m Message) error {
	e.waitQ.require(a) <- &m
	pm := <-e.recvQ.require(a)
	if !m.same(pm) {
		return errRegisteredBufferNotUsed
	}
	return nil
}

func (e *CollectiveEndpoint) accept(conn net.Conn, remote plan.NetAddr) (string, *Message, error) {
	var mh messageHeader
	if err := mh.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	name := string(mh.Name)
	if mh.HasFlag(WaitRecvBuf) {
		m := <-e.waitQ.require(remote.WithName(name))
		if err := m.ReadInto(conn); err != nil {
			return "", nil, err
		}
		return name, m, nil
	}
	var m Message
	if err := m.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	return name, &m, nil
}

func (e *CollectiveEndpoint) handle(name string, msg *Message, conn net.Conn, remote plan.NetAddr) {
	e.recvQ.require(remote.WithName(name)) <- msg
}
