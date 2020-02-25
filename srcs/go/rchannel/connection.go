package rchannel

import (
	"errors"
	"io"
	"net"
	"sync"
	"time"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

// Connection is a simplex logical connection from one peer to another
type Connection interface {
	io.Closer
	Send(msgName string, m Message, flags uint32) error
	Read(msgName string, m Message) error
}

func NewPingConnection(remote, local plan.NetAddr) Connection {
	c := newConnection(remote, local, ConnPing, 0) // FIXME: use token
	if err := c.initOnce(); err != nil {
		log.Errorf("ping connection initOnce failed: %v", err)
	}
	return c
}

var errInvalidToken = errors.New("invalid token")

func newConnection(remote, local plan.NetAddr, t ConnType, token uint32) *tcpConnection {
	init := func() (net.Conn, error) {
		conn, err := func() (net.Conn, error) {
			if kc.UseUnixSock && remote.ColocatedWith(local) {
				addr := net.UnixAddr{Name: remote.SockFile(), Net: "unix"}
				return net.DialUnix(addr.Net, nil, &addr)
			}
			return net.Dial("tcp", remote.String())
		}()
		if err != nil {
			return nil, err
		}
		h := connectionHeader{
			Type:    uint16(t),
			SrcIPv4: local.IPv4,
			SrcPort: local.Port,
		}
		if err := h.WriteTo(conn); err != nil {
			return nil, err
		}
		var ack connectionACK
		if err := ack.ReadFrom(conn); err != nil {
			return nil, err
		}
		if ack.Token != token {
			if t == ConnCollective {
				conn.Close()
				return nil, errInvalidToken
			}
			// FIXME: ignored token check for other connection types
		}
		return conn, nil
	}
	return &tcpConnection{remote: remote, init: init}
}

type tcpConnection struct {
	sync.Mutex
	remote plan.NetAddr
	init   func() (net.Conn, error)
	conn   net.Conn
}

var errCantEstablishConnection = errors.New("can't establish connection")

func (c *tcpConnection) initOnce() error {
	c.Lock()
	defer c.Unlock()
	if c.conn != nil {
		return nil
	}
	t0 := time.Now()
	for i := 0; i <= kc.ConnRetryCount; i++ {
		var err error
		if c.conn, err = c.init(); err == nil {
			log.Debugf("connection to #<%s> established after %d trials, took %s", c.remote, i+1, time.Since(t0))
			return nil
		}
		log.Debugf("failed to establish connection to #<%s> for %d times: %v", c.remote, i+1, err)
		time.Sleep(kc.ConnRetryPeriod)
	}
	return errCantEstablishConnection
}

func (c *tcpConnection) Send(msgName string, m Message, flags uint32) error {
	c.initOnce()
	c.Lock()
	defer c.Unlock()
	bs := []byte(msgName)
	mh := messageHeader{
		NameLength: uint32(len(bs)),
		Name:       bs,
		Flags:      flags,
	}
	if err := mh.WriteTo(c.conn); err != nil {
		return err
	}
	return m.WriteTo(c.conn)
}

func (c *tcpConnection) Read(msgName string, m Message) error {
	c.initOnce()
	c.Lock()
	defer c.Unlock()
	var mh messageHeader
	if err := mh.ReadFromLike(c.conn, msgName); err != nil {
		return err
	}
	return m.ReadInto(c.conn)
}

func (c *tcpConnection) Close() error {
	c.Lock()
	defer c.Unlock()
	return c.conn.Close()
}
