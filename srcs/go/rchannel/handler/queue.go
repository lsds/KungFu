package handler

import (
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

// QueueHandler handles queue connection
type QueueHandler struct {
	qs *BufferPool
}

func NewQueueHandler() *QueueHandler {
	return &QueueHandler{
		qs: newBufferPool(1),
	}
}

func (h *QueueHandler) Get(peer plan.PeerID, name string) *connection.Message {
	m := <-h.qs.require(peer.WithName(name))
	return m
}

func (h *QueueHandler) accept(conn connection.Connection) (string, *connection.Message, error) {
	var mh connection.MessageHeader
	if err := mh.ReadFrom(conn.Conn()); err != nil {
		return "", nil, err
	}
	name := string(mh.Name)
	var m connection.Message
	m.Flags = mh.Flags
	if err := m.ReadFrom(conn.Conn()); err != nil {
		return "", nil, err
	}
	return name, &m, nil
}

func (h *QueueHandler) handle(name string, msg *connection.Message, conn connection.Connection) {
	log.Errorf(`got new item from queue[%s] %d bytes`, name, msg.Length)
	// TODO: ACK
	h.qs.require(conn.Src().WithName(name)) <- msg
}

func (h *QueueHandler) Handle(conn connection.Connection) (int, error) {
	return connection.Stream(conn, h.accept, h.handle)
}
