package peer

import (
	"context"
	"errors"

	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/client"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/rchannel/handler"
)

type router struct {
	self        plan.PeerID
	Collective  *handler.CollectiveEndpoint
	P2P         *handler.PeerToPeerEndpoint
	ctrlHandler *handler.ControlHandler
	pingHandler *handler.PingHandler
	client      *client.Client
}

func NewRouter(self plan.PeerID) *router {
	client := client.New(self, config.UseUnixSock)
	router := &router{
		self:        self,
		Collective:  handler.NewCollectiveEndpoint(),
		P2P:         handler.NewPeerToPeerEndpoint(client),
		ctrlHandler: &handler.ControlHandler{},
		pingHandler: &handler.PingHandler{},
		client:      client,
	}
	return router
}

func (r *router) Self() plan.PeerID {
	return r.self
}

func (r *router) ResetConnections(keeps plan.PeerList, token uint32) {
	r.client.ResetConnections(keeps, token)
}

// Send sends data in buf to given Addr
func (r *router) Send(a plan.Addr, buf []byte, t connection.ConnType, flags uint32) error {
	return r.client.Send(a, buf, t, flags)
}

var errWaitPeerFailed = errors.New("wait peer failed")

func (r *router) Wait(ctx context.Context, target plan.PeerID) (int, error) {
	n, ok := r.client.Wait(ctx, target)
	if !ok {
		return n, errWaitPeerFailed
	}
	return n, nil
}

// Handle implements Handle method of ConnHandler interface
func (r *router) Handle(conn connection.Connection) (int, error) {
	switch t := conn.Type(); t {
	case connection.ConnCollective:
		return r.Collective.Handle(conn)
	case connection.ConnPeerToPeer:
		return r.P2P.Handle(conn)
	case connection.ConnControl:
		return r.ctrlHandler.Handle(conn)
	case connection.ConnPing:
		return r.pingHandler.Handle(conn)
	default:
		return 0, connection.ErrInvalidConnectionType
	}
}
