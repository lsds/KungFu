package client

import (
	"sync"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

type connKey struct {
	a plan.PeerID
	t connection.ConnType
}

type connectionPool struct {
	sync.Mutex
	useUnixSock bool
	conns       map[connKey]connection.Connection
	token       uint32
}

func newConnectionPool(useUnixSock bool) *connectionPool {
	return &connectionPool{
		useUnixSock: useUnixSock,
		conns:       make(map[connKey]connection.Connection),
	}
}

func (p *connectionPool) get(remote, local plan.PeerID, t connection.ConnType) connection.Connection {
	p.Lock()
	defer p.Unlock()
	key := connKey{remote, t}
	if conn, ok := p.conns[key]; ok {
		return conn
	}
	conn := connection.New(remote, local, t, p.token, p.useUnixSock)
	p.conns[key] = conn
	return conn
}

func (p *connectionPool) reset(keeps plan.PeerList, token uint32) {
	m := keeps.Set()
	p.Lock()
	defer p.Unlock()
	p.token = token
	for k := range p.conns {
		if _, ok := m[plan.PeerID(k.a)]; !ok {
			delete(p.conns, k) // FIXME: gracefully shutdown conn
		}
	}
}
