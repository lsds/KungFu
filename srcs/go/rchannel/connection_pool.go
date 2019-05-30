package rchannel

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
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
		connRetryPeriod:   1500 * time.Millisecond,
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
			// log.Infof("Return connection to %s.", remote.String())
			return conn, nil
		}
		if time.Since(t0) > p.connRetryDuration {
			break
		}
		<-tk.C
	}

	log.Warnf("Fail to get connection.")
	return nil, errCantEstablishConnection
}
