package rchannel

import (
	"io"
	"net"
	"sync"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/shmpool"
)

// Connection is a simplex logical connection from one peer to another
type Connection interface {
	io.Closer
	Send(name string, m Message) error
}

func parseIPv4(host string) uint32 {
	ip := net.ParseIP(host).To4()
	a := uint32(ip[0]) << 24
	b := uint32(ip[1]) << 16
	c := uint32(ip[2]) << 8
	d := uint32(ip[3])
	return a | b | c | d
}

func newConnection(remote, local plan.NetAddr, t ConnType) (Connection, error) {
	conn, err := func() (net.Conn, error) {
		if remote.Host == local.Host {
			addr := net.UnixAddr{remote.SockFile(), "unix"}
			return net.DialUnix(addr.Net, nil, &addr)
		}
		return net.Dial("tcp", remote.String())
	}()
	if err != nil {
		return nil, err
	}
	h := connectionHeader{
		Type:    uint16(t),
		SrcIPv4: parseIPv4(local.Host), // parseIPv4 :: str -> uint32
		SrcPort: local.Port,
	}
	if err := h.WriteTo(conn); err != nil {
		return nil, err
	}
	tc := tcpConnection{conn: conn}
	if kc.UseShm && remote.Host == local.Host {
		pool, err := shmpool.New(plan.ShmNameFor(local, remote))
		if err != nil {
			return nil, err
		}
		sc := &shmConnection{
			tcpConnection: tc,
			pool:          pool,
		}
		go sc.handleAck()
		return sc, nil
	}
	return &tc, nil
}

type tcpConnection struct {
	sync.Mutex
	conn net.Conn
}

func (c *tcpConnection) Send(name string, m Message) error {
	c.Lock()
	defer c.Unlock()
	bs := []byte(name)
	mh := messageHeader{
		NameLength: uint32(len(bs)),
		Name:       bs,
	}
	if err := mh.WriteTo(c.conn); err != nil {
		return err
	}
	return m.WriteTo(c.conn)
}

func (c *tcpConnection) Close() error {
	return c.conn.Close()
}

type shmConnection struct {
	tcpConnection
	pool *shmpool.Pool
}

func (c *shmConnection) handleAck() error {
	for {
		var mt messageTail
		if err := mt.ReadFrom(c.conn); err != nil {
			return err
		}
		b := shmpool.Block{Offset: int(mt.Offset), Size: int(mt.Length)}
		c.pool.Put(b)
	}
}

func (c *shmConnection) Send(name string, m Message) error {
	c.Lock()
	defer c.Unlock()
	bs := []byte(name)
	mh := messageHeader{
		NameLength: uint32(len(bs)),
		Name:       bs,
		BodyInShm:  1,
	}
	if err := mh.WriteTo(c.conn); err != nil {
		return err
	}
	b := c.pool.Get(int(m.Length))
	c.pool.Write(b, m.Data)
	mt := messageTail{
		Offset: uint32(b.Offset),
		Length: m.Length,
	}
	return mt.WriteTo(c.conn)
}
