package rchannel

import (
	"net"
	"os"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/shm"
)

type Router struct {
	localAddr  plan.NetAddr
	bufferPool *BufferPool
	connPool   *ConnectionPool
	metrics    *monitor.NetMetrics
}

func NewRouter(self plan.PeerSpec) *Router {
	return &Router{
		localAddr:  self.NetAddr,
		bufferPool: newBufferPool(),     // in-comming messages
		connPool:   newConnectionPool(), // out-going connections
		metrics:    monitor.GetNetMetrics(),
	}
}

// getChannel returns the Channel of given Addr
func (r *Router) getChannel(a plan.Addr) (*Channel, error) {
	conn, err := r.connPool.get(a.NetAddr(), r.localAddr)
	if err != nil {
		return nil, err
	}
	return newChannel(a.Name, conn), nil
}

// Send sends data in buf to given Addr
func (r *Router) Send(a plan.Addr, buf []byte) error {
	m := Message{
		Length: uint32(len(buf)),
		Data:   buf,
	}
	if err := r.send(a, m); err != nil {
		log.Errorf("Router::Send failed: %v", err)
		// TODO: retry
		os.Exit(1)
		// return err
	}
	r.metrics.Sent(int64(m.Length))
	return nil
}

func (r *Router) send(a plan.Addr, m Message) error {
	// log.Infof("%s::%s", "Router", "Send")
	ch, err := r.getChannel(a)
	if err != nil {
		return err
	}
	if err := ch.Send(m); err != nil {
		return err
	}
	return nil
}

// Recv recevies a message from given Addr
func (r *Router) Recv(a plan.Addr) Message {
	// log.Infof("%s::%s(%s)", "Router", "Recv", a)
	// TODO: reduce memory copy
	m := *<-r.bufferPool.require(a)
	// TODO: add timeout
	return m
}

func (r *Router) acceptOne(conn net.Conn, shm shm.Shm) (string, *Message, error) {
	var mh messageHeader
	if err := mh.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	var m Message
	if mh.BodyInShm != 0 {
		var mt messageTail
		if err := mt.ReadFrom(conn); err != nil {
			return "", nil, err
		}
		m.Length = mt.Length
		m.Data = make([]byte, m.Length)
		shm.Seek(int(mt.Offset))
		shm.Read(m.Data, int(m.Length))
		mt.WriteTo(conn)
	} else {
		if err := m.ReadFrom(conn); err != nil {
			return "", nil, err
		}
	}
	return string(mh.Name), &m, nil
}

var newShm = shm.New

func (r *Router) stream(conn net.Conn, remote plan.NetAddr) (int, error) {
	var shm shm.Shm
	if kc.UseShm && remote.Host == r.localAddr.Host {
		var err error
		if shm, err = newShm(plan.ShmNameFor(remote, r.localAddr)); err != nil {
			return 0, err
		}
		defer shm.Close()
	}
	for i := 0; ; i++ {
		name, m, err := r.acceptOne(conn, shm)
		if err != nil {
			return i, err
		}
		r.metrics.Recv(int64(m.Length))
		r.bufferPool.require(remote.WithName(name)) <- m
	}
}
