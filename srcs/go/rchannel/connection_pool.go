package rchannel

import (
	"errors"
	"log"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
)

var errCantEstablishConnection = errors.New("can't establish connection")

type ConnectionPool struct {
	sync.Mutex
	conns map[string]Connection

	connRetryDuration time.Duration
	connRetryPeriod   time.Duration
}

func newConnectionPool() *ConnectionPool {
	return &ConnectionPool{
		conns: make(map[string]Connection),

		connRetryDuration: 120 * time.Second,
		connRetryPeriod:   500 * time.Millisecond,
	}
}

func (p *ConnectionPool) get(remote, local plan.NetAddr) (Connection, error) {
	p.Lock()
	defer p.Unlock()
	tk := time.NewTicker(p.connRetryPeriod)
	defer tk.Stop()
	t0 := time.Now()
	for i := 0; ; i++ {
		if conn, ok := p.conns[remote.String()]; !ok {
			log.Printf("%d-th attempt to connect to %s", i, remote.String())
			conn, err := newConnection(remote, local)
			if err == nil {
				p.conns[remote.String()] = conn
			} else {
				log.Printf("newConnection failed: %v", err)
			}
		} else {
			log.Printf("Return connection to %s.", remote.String())
			return conn, nil
		}
		if time.Since(t0) > p.connRetryDuration {
			break
		}
		<-tk.C
	}

	log.Printf("Fail to get connection.")
	return nil, errCantEstablishConnection
}
