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
}

func newConnectionPool() *ConnectionPool {
	return &ConnectionPool{
		conns: make(map[string]*Connection),
	}
}

func (p *ConnectionPool) get(netAddr string, localPort uint32) (*Connection, error) {
	p.Lock()
	defer p.Unlock()

	tk := time.NewTicker(100 * time.Millisecond)
	defer tk.Stop()

	trials := 10
	for i := 0; i <= trials; i++ {
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
