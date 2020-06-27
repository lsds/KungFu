package handler

import (
	"errors"
	"sync"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/shm"
)

type CollectiveEndpoint struct {
	self  plan.PeerID
	waitQ *BufferPool
	recvQ *BufferPool

	shms  map[plan.PeerID]shm.Shm
	shmMu sync.Mutex
}

var newShm = shm.New

func NewCollectiveEndpoint(self plan.PeerID) *CollectiveEndpoint {
	return &CollectiveEndpoint{
		self:  self,
		waitQ: newBufferPool(1),
		recvQ: newBufferPool(1),
		shms:  make(map[plan.PeerID]shm.Shm),
	}
}

func (e *CollectiveEndpoint) getSHMFrom(a plan.PeerID) shm.Shm {
	e.shmMu.Lock()
	defer e.shmMu.Unlock()
	if _, ok := e.shms[a]; !ok {
		shm, err := newShm(a.SHMNameTo(e.self))
		if err != nil {
			utils.ExitErr(err)
		}
		e.shms[a] = shm
	}
	return e.shms[a]
}

// Handle implements ConnHandler.Handle interface
func (e *CollectiveEndpoint) Handle(conn connection.Connection) (int, error) {
	return connection.Stream(conn, e.accept, e.handle)
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

func (e *CollectiveEndpoint) accept(conn connection.Connection) (string, *connection.Message, error) {
	var mh connection.MessageHeader
	if err := mh.ReadFrom(conn.Conn()); err != nil {
		return "", nil, err
	}
	name := string(mh.Name)
	if mh.HasFlag(connection.BodyInSHM) {
		// log.Errorf("reading from SHM")
		var mt connection.MessageTail
		if err := mt.ReadFrom(conn.Conn()); err != nil {
			return "", nil, err
		}
		// log.Errorf("got %#v", mt)
		var m *connection.Message
		if mh.HasFlag(connection.WaitRecvBuf) {
			m = <-e.waitQ.require(conn.Src().WithName(name))
		} else {
			m = &connection.Message{
				Length: mt.Length,
				Data:   connection.GetBuf(mt.Length),
			}
		}
		shm := e.getSHMFrom(conn.Src())
		shm.Seek(int(mt.Offset))
		shm.Read(m.Data, int(m.Length))
		if err := mt.WriteTo(conn.Conn()); err != nil {
			return "", nil, err
		}
		return name, m, nil
	}
	if mh.HasFlag(connection.WaitRecvBuf) {
		m := <-e.waitQ.require(conn.Src().WithName(name))
		if err := m.ReadInto(conn.Conn()); err != nil {
			return "", nil, err
		}
		return name, m, nil
	}
	var m connection.Message
	if err := m.ReadFrom(conn.Conn()); err != nil {
		return "", nil, err
	}
	return name, &m, nil
}

func (e *CollectiveEndpoint) handle(name string, msg *connection.Message, conn connection.Connection) {
	e.recvQ.require(conn.Src().WithName(name)) <- msg
}
