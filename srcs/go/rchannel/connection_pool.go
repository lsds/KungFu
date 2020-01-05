package rchannel

import (
	"sync"

	"github.com/lsds/KungFu/srcs/go/plan"
)

type connKey struct {
	a plan.NetAddr
	t ConnType
}

type ConnectionPool struct {
	sync.Mutex
	conns map[connKey]Connection
}

func newConnectionPool() *ConnectionPool {
	return &ConnectionPool{
		conns: make(map[connKey]Connection),
	}
}

func (p *ConnectionPool) get(remote, local plan.NetAddr, t ConnType) Connection {
	p.Lock()
	defer p.Unlock()
	key := connKey{remote, t}
	if conn, ok := p.conns[key]; ok {
		return conn
	}
	conn := newConnection(remote, local, t)
	p.conns[key] = conn
	return conn
}

func (p *ConnectionPool) reset(keeps plan.PeerList) {
	m := keeps.Set()
	p.Lock()
	defer p.Unlock()
	for k := range p.conns {
		if _, ok := m[plan.PeerID(k.a)]; !ok {
			delete(p.conns, k) // FIXME: gracefully shutdown conn
		}
	}
}
