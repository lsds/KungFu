package rchannel

import (
	"net"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/client"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

type Router struct {
	self       plan.PeerID
	Collective *CollectiveEndpoint // FIXME: move it out of Router
	P2P        *PeerToPeerEndpoint
	client     *client.Client
}

func NewRouter(self plan.PeerID) *Router {
	router := &Router{
		self:       self,
		Collective: NewCollectiveEndpoint(),
		client:     client.New(self),
	}
	router.P2P = NewPeerToPeerEndpoint(router) // FIXME: remove mutual membership
	return router
}

func (r *Router) Self() plan.PeerID {
	return r.self
}

func (r *Router) ResetConnections(keeps plan.PeerList, token uint32) {
	r.client.ResetConnections(keeps, token)
}

// Send sends data in buf to given Addr
func (r *Router) Send(a plan.Addr, buf []byte, t connection.ConnType, flags uint32) error {
	return r.client.Send(a, buf, t, flags)
}

// Handle implements Handle method of ConnHandler interface
func (r *Router) Handle(conn net.Conn, remote plan.NetAddr, t connection.ConnType) error {
	switch t {
	case connection.ConnCollective:
		return r.Collective.Handle(conn, remote, t)
	case connection.ConnPeerToPeer:
		return r.P2P.Handle(conn, remote, t)
	case connection.ConnControl:
		var h controlHandler
		return h.Handle(conn, remote, t)
	default:
		return connection.ErrInvalidConnectionType
	}
}
