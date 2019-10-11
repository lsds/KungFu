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
	var lastErr error
	var mu sync.Mutex
	{
		tk := time.NewTicker(2 * time.Second)
		defer tk.Stop()
		go func() {
			for range tk.C {
				mu.Lock()
				log.Warnf("still trying to connect to %s, last error: %v", remote, lastErr)
				mu.Unlock()
			}
		}()
	}
	{
		tk := time.NewTicker(p.connRetryPeriod)
		defer tk.Stop()
		for i := 0; i <= p.connRetryCount; i++ {
			if conn, ok := p.conns[connKey{remote, t}]; !ok {
				conn, err := newConnection(remote, local, t)
				if err == nil {
					p.conns[connKey{remote, t}] = conn
					return conn, nil
				}
				mu.Lock()
				lastErr = err
				mu.Unlock()
			} else {
				return conn, nil
			}
			<-tk.C
		}
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
