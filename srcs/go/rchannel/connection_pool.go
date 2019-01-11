package rchannel

import (
	"errors"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
)

var errCantEstablishConnection = errors.New("can't establish connection")

type ConnectionPool struct {
	sync.Mutex
	conns map[string]*Connection

	connRetryCount  int
	connRetryPeriod time.Duration
}

func newConnectionPool() *ConnectionPool {
	return &ConnectionPool{
		conns: make(map[string]*Connection),

		connRetryCount:  40,
		connRetryPeriod: 500 * time.Millisecond,
	}
}

func (p *ConnectionPool) get(a plan.NetAddr, localHost string, localPort uint32) (*Connection, error) {
	p.Lock()
	defer p.Unlock()
	tk := time.NewTicker(p.connRetryPeriod)
	defer tk.Stop()
	for i := 0; i <= p.connRetryCount; i++ {
		if conn, ok := p.conns[a.String()]; !ok {
			conn, err := newConnection(a, localHost, localPort)
			if err == nil {
				p.conns[a.String()] = conn
			}
		} else {
			return conn, nil
		}
		<-tk.C
	}
	return nil, errCantEstablishConnection
}
