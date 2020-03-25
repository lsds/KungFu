package connection

import (
	"errors"
	"io"
	"net"
	"sync"
	"time"

	kc "github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

// Connection is a simplex logical connection from one peer to another
type Connection interface {
	io.Closer
	Send(msgName string, m Message, flags uint32) error
	Read(msgName string, m Message) error
}

func NewPingConnection(remote, local plan.PeerID) (Connection, error) {
	return newPingConnection(remote, local)
}

func newPingConnection(remote, local plan.PeerID) (Connection, error) {
	c := newConnection(remote, local, ConnPing, 0) // FIXME: use token
	if err := c.initOnce(); err != nil {
		return nil, err
	}
	return c, nil
}

var errInvalidToken = errors.New("invalid token")

func NewConnection(remote, local plan.PeerID, t ConnType, token uint32) *tcpConnection {
	return newConnection(remote, local, t, token)
}

func newConnection(remote, local plan.PeerID, t ConnType, token uint32) *tcpConnection {
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
		h := ConnectionHeader{
			Type:    uint16(t),
			SrcIPv4: local.IPv4,
			SrcPort: local.Port,
		}
		if err := h.WriteTo(conn); err != nil {
			return nil, err
		}
		var ack ConnectionACK
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
	var initRetry int
	if t == ConnCollective || t == ConnPeerToPeer {
		initRetry = kc.ConnRetryCount
	}
	return &tcpConnection{
		remote:    remote,
		init:      init,
		initRetry: initRetry,
		connType:  t,
	}
}

type tcpConnection struct {
	sync.Mutex
	remote    plan.PeerID
	init      func() (net.Conn, error)
	conn      net.Conn
	initRetry int
	connType  ConnType
}

var errCantEstablishConnection = errors.New("can't establish connection")

func (c *tcpConnection) initOnce() error {
	c.Lock()
	defer c.Unlock()
	if c.conn != nil {
		return nil
	}
	t0 := time.Now()
	for i := 0; i <= c.initRetry; i++ {
		var err error
		if c.conn, err = c.init(); err == nil {
			log.Debugf("%s connection to #<%s> established after %d trials, took %s", c.connType, c.remote, i+1, time.Since(t0))
			return nil
		}
		log.Debugf("failed to establish connection to #<%s> for %d times: %v", c.remote, i+1, err)
		time.Sleep(kc.ConnRetryPeriod)
	}
	return errCantEstablishConnection
}

func (c *tcpConnection) Send(msgName string, m Message, flags uint32) error {
	if err := c.initOnce(); err != nil {
		return err
	}
	c.Lock()
	defer c.Unlock()
	bs := []byte(msgName)
	mh := MessageHeader{
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
	if err := c.initOnce(); err != nil {
		return err
	}
	c.Lock()
	defer c.Unlock()
	var mh MessageHeader
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
