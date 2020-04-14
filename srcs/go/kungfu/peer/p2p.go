package peer

import (
	"errors"

	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

var (
	errInvalidRank = errors.New("invalid rank")
)

func (p *Peer) SaveVersion(version, name string, buf *base.Vector) error {
	return p.router.P2P.SaveVersion(version, name, buf)
}

func (p *Peer) Save(name string, buf *base.Vector) error {
	return p.router.P2P.Save(name, buf)
}

func (p *Peer) Request(target plan.PeerID, version, name string, buf *base.Vector) (bool, error) {
	return p.router.P2P.Request(target.WithName(name), version, asMessage(buf))
}

func (p *Peer) RequestRank(rank int, version, name string, buf *base.Vector) (bool, error) {
	sess := p.CurrentSession()
	if rank < 0 || sess.Size() <= rank {
		return false, errInvalidRank
	}
	target := sess.Peer(rank)
	return p.Request(target, version, name, buf)
}

func asMessage(b *base.Vector) connection.Message {
	return connection.Message{
		Length: uint32(len(b.Data)),
		Data:   b.Data,
	}
}
