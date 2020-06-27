package connection

import (
	"errors"
	"io"
	"net"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/shmpool"
)

// Connection is a simplex logical connection from one peer to another
type Connection interface {
	io.Closer

	Conn() net.Conn // FIXME: don't allow access net.Conn
	Type() ConnType
	Src() plan.PeerID
	Dest() plan.PeerID
	Send(name string, m Message, flags uint32) error
	Read(name string, m Message) error
}

// UpgradeFrom performs the server side operations to upgrade a TCP connection to a Connection
func UpgradeFrom(conn net.Conn, self plan.PeerID, token uint32) (Connection, error) {
	var ch connectionHeader
	if err := ch.ReadFrom(conn); err != nil {
		return nil, err
	}
	ack := connectionACK{
		Token: token,
	}
	if err := ack.WriteTo(conn); err != nil {
		return nil, err
	}
	return &tcpConnection{
		src:      plan.PeerID{IPv4: ch.SrcIPv4, Port: ch.SrcPort},
		dest:     self,
		connType: ConnType(ch.Type),
		conn:     conn,
	}, nil
}

var errInvalidToken = errors.New("invalid token")

func Open(remote, local plan.PeerID, t ConnType, token uint32, useUnixSock bool) (*tcpConnection, error) {
	conn := New(remote, local, t, token, useUnixSock)
	if err := conn.initOnce(); err != nil {
		return nil, err
	}
	return conn, nil
}

func New(remote, local plan.PeerID, t ConnType, token uint32, useUnixSock bool) *tcpConnection {
	init := func() (net.Conn, error) {
		conn, err := func() (net.Conn, error) {
			if useUnixSock && remote.ColocatedWith(local) {
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
	var initRetry int
	if t == ConnCollective || t == ConnPeerToPeer {
		initRetry = config.ConnRetryCount
	}
	var pool *shmpool.Pool
	if t == ConnCollective && config.UseSHM {
		var err error
		if pool, err = shmpool.New(local.SHMNameTo(remote)); err != nil {
			utils.ExitErr(err)
		}
	}
	return &tcpConnection{
		init:      init,
		src:       local,
		dest:      remote,
		initRetry: initRetry,
		connType:  t,
		pool:      pool,
	}
}

type tcpConnection struct {
	sync.Mutex
	src, dest plan.PeerID
	init      func() (net.Conn, error)
	conn      net.Conn
	initRetry int
	connType  ConnType
	pool      *shmpool.Pool
}

var errCantEstablishConnection = errors.New("can't establish connection")

func (c *tcpConnection) Conn() net.Conn {
	return c.conn
}

func (c *tcpConnection) Type() ConnType {
	return c.connType
}

func (c *tcpConnection) Src() plan.PeerID {
	return c.src
}

func (c *tcpConnection) Dest() plan.PeerID {
	return c.dest
}

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
			log.Debugf("%s connection to #<%s> established after %d trials, took %s", c.connType, c.dest, i+1, time.Since(t0))
			if c.pool != nil {
				go c.handleSHMAck()
			}
			return nil
		}
		log.Debugf("failed to establish connection to #<%s> for %d times: %v", c.dest, i+1, err)
		time.Sleep(config.ConnRetryPeriod)
	}
	return errCantEstablishConnection
}

func (c *tcpConnection) Send(name string, m Message, flags uint32) error {
	usingSHM := config.UseSHM && m.Length >= config.SHMThreshold
	if usingSHM {
		// log.Errorf("using SHM")
		flags |= BodyInSHM
	}
	if err := c.initOnce(); err != nil {
		return err
	}
	c.Lock()
	defer c.Unlock()
	bs := []byte(name)
	mh := MessageHeader{
		NameLength: uint32(len(bs)),
		Name:       bs,
		Flags:      flags,
	}
	if err := mh.WriteTo(c.conn); err != nil {
		return err
	}
	if usingSHM {
		b := c.pool.Get(int(m.Length))
		c.pool.Write(b, m.Data)
		mt := MessageTail{
			Offset: uint32(b.Offset),
			Length: m.Length,
		}
		return mt.WriteTo(c.conn)
	}
	return m.WriteTo(c.conn)
}

func (c *tcpConnection) Read(name string, m Message) error {
	if err := c.initOnce(); err != nil {
		return err
	}
	c.Lock()
	defer c.Unlock()
	var mh MessageHeader
	if err := mh.Expect(c.conn, name); err != nil {
		return err
	}
	return m.ReadInto(c.conn)
}

func (c *tcpConnection) handleSHMAck() error {
	for {
		var mt MessageTail
		if err := mt.ReadFrom(c.conn); err != nil {
			return err
		}
		b := shmpool.Block{Offset: int(mt.Offset), Size: int(mt.Length)}
		c.pool.Put(b)
	}
}

func (c *tcpConnection) Close() error {
	c.Lock()
	defer c.Unlock()
	return c.conn.Close()
}
