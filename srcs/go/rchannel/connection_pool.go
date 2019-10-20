package rchannel

import (
	"errors"
	"sync"
	"time"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

var errCantEstablishConnection = errors.New("can't establish connection")

type connKey struct {
	a plan.NetAddr
	t ConnType
}

type ConnectionPool struct {
	sync.Mutex
	conns map[connKey]Connection

	connRetryCount  int
	connRetryPeriod time.Duration
}

func newConnectionPool() *ConnectionPool {
	return &ConnectionPool{
		conns: make(map[connKey]Connection),

		connRetryCount:  kc.ConnRetryCount,
		connRetryPeriod: kc.ConnRetryPeriod,
	}
}

func (p *ConnectionPool) get(remote, local plan.NetAddr, t ConnType) (Connection, error) {
	p.Lock()
	defer p.Unlock()
	key := connKey{remote, t}
	if conn, ok := p.conns[key]; ok {
		return conn, nil
	}

	log.Debugf("New connection to %s", remote)
	for i := 0; i <= p.connRetryCount; i++ {
		// TODO: call newConnection with timeout.
		conn, err := newConnection(remote, local, t)
		if err == nil {
			p.conns[key] = conn
			return conn, nil
		}

		log.Warnf("Retry connect to [%s] for [%d] times. Retry after %s. Error: %v. ", remote, i+1, p.connRetryPeriod, err)
		time.Sleep(p.connRetryPeriod)
	}

	return nil, errCantEstablishConnection
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
