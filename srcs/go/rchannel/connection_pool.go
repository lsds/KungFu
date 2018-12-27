package rchannel

import (
	"errors"
	"sync"
	"time"
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

func (p *ConnectionPool) get(netAddr string, localPort uint32) (*Connection, error) {
	p.Lock()
	defer p.Unlock()

	tk := time.NewTicker(p.connRetryPeriod)
	defer tk.Stop()

	for i := 0; i <= p.connRetryCount; i++ {
		if conn, ok := p.conns[netAddr]; !ok {
			conn, err := newConnection(netAddr, localPort)
			if err == nil {
				p.conns[netAddr] = conn
			}
		} else {
			return conn, nil
		}
		<-tk.C
	}
	return nil, errCantEstablishConnection
}
