package rchannel

import (
	"errors"
	"net"

	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type Router struct {
	localAddr  plan.NetAddr
	Collective *CollectiveEndpoint // FIXME: move it out of Router
	P2P        *PeerToPeerEndpoint
	connPool   *ConnectionPool
	monitor    monitor.Monitor
}

func NewRouter(self plan.PeerID) *Router {
	router := &Router{
		localAddr:  plan.NetAddr(self),
		Collective: NewCollectiveEndpoint(),
		connPool:   newConnectionPool(), // out-going connections
		monitor:    monitor.GetMonitor(),
	}
	router.P2P = NewPeerToPeerEndpoint(router) // FIXME: remove mutual membership
	return router
}

func (r *Router) Self() plan.PeerID {
	return plan.PeerID(r.localAddr)
}

func (r *Router) ResetConnections(keeps plan.PeerList, token uint32) {
	r.connPool.reset(keeps, token)
}

// Send sends data in buf to given Addr
func (r *Router) Send(a plan.Addr, buf []byte, t ConnType, flags uint32) error {
	msg := Message{
		Length: uint32(len(buf)),
		Data:   buf,
	}
	if err := r.send(a, msg, t, flags); err != nil {
		return err
	}
	r.monitor.Egress(int64(msg.Length), a.NetAddr())
	return nil
}

func (r *Router) send(a plan.Addr, msg Message, t ConnType, flags uint32) error {
	conn := r.connPool.get(a.NetAddr(), r.localAddr, t)
	if err := conn.Send(a.Name, msg, flags); err != nil {
		return err
	}
	return nil
}

var ErrInvalidConnectionType = errors.New("invalid connection type")

// Handle implements ConnHandler.Handle interface
func (r *Router) Handle(conn net.Conn, remote plan.NetAddr, t ConnType) error {
	switch t {
	case ConnCollective:
		return r.Collective.Handle(conn, remote, t)
	case ConnPeerToPeer:
		return r.P2P.Handle(conn, remote, t)
	default:
		return ErrInvalidConnectionType
	}
}
