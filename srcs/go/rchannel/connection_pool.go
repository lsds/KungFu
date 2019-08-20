package rchannel

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
)

var errCantEstablishConnection = errors.New("can't establish connection")

type ConnectionPool struct {
	sync.Mutex
	conns map[string]Connection

	connRetryCount  int
	connRetryPeriod time.Duration
}

func newConnectionPool() *ConnectionPool {
	return &ConnectionPool{
		conns: make(map[string]Connection),

		connRetryCount:  200,
		connRetryPeriod: 500 * time.Millisecond,
	}
}

func (p *ConnectionPool) get(remote, local plan.NetAddr, t ConnType) (Connection, error) {
	p.Lock()
	defer p.Unlock()
	tk := time.NewTicker(p.connRetryPeriod)
	defer tk.Stop()
	key := func(a plan.NetAddr, t ConnType) string {
		return fmt.Sprintf("%s#%d", a, t)
	}
	for i := 0; i <= p.connRetryCount; i++ {
		if conn, ok := p.conns[key(remote, t)]; !ok {
			conn, err := newConnection(remote, local, t)
			if err == nil {
				p.conns[key(remote, t)] = conn
			}
		} else {
			return conn, nil
		}
		<-tk.C
	}
	return nil, errCantEstablishConnection
}

func (p *ConnectionPool) reset() {
	p.Lock()
	defer p.Unlock()
	for k := range p.conns {
		delete(p.conns, k) // FIXME: gracefully shutdown conn
	}
}
