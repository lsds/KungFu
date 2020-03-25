package rchannel

import (
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
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
	conn, err := newPingConnection(plan.NetAddr(target), plan.NetAddr(c.self))
	if err != nil {
		return time.Since(t0), err
	}
	defer conn.Close()
	var empty Message
	if err := conn.Send("ping", empty, NoFlag); err != nil {
		return time.Since(t0), err
	}
	if err := conn.Read("ping", empty); err != nil {
		return time.Since(t0), err
	}
	return time.Since(t0), nil
}
