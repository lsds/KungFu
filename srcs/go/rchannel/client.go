package rchannel

import (
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

type Client struct {
	self plan.PeerID
}

func NewClient(self plan.PeerID) *Client {
	return &Client{
		self: self,
	}
}

func (c *Client) Ping(target plan.PeerID) (time.Duration, error) {
	t0 := time.Now()
	conn, err := connection.NewPingConnection(target, c.self)
	if err != nil {
		return time.Since(t0), err
	}
	defer conn.Close()
	var empty connection.Message
	if err := conn.Send("ping", empty, connection.NoFlag); err != nil {
		return time.Since(t0), err
	}
	if err := conn.Read("ping", empty); err != nil {
		return time.Since(t0), err
	}
	return time.Since(t0), nil
}
