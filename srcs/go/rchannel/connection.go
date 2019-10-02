package rchannel

import (
	"io"
	"net"
	"sync"

	"github.com/lsds/KungFu/srcs/go/plan"
)

// Connection is a simplex logical connection from one peer to another
type Connection interface {
	io.Closer
	Send(msgName string, m Message, flags uint32) error
	Read(msgName string, m Message) error
}

func parseIPv4(host string) uint32 {
	ip := net.ParseIP(host).To4()
	a := uint32(ip[0]) << 24
	b := uint32(ip[1]) << 16
	c := uint32(ip[2]) << 8
	d := uint32(ip[3])
	return a | b | c | d
}

func NewPingConnection(remote, local plan.NetAddr) (Connection, error) {
	return newConnection(remote, local, ConnPing)
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
	return &tcpConnection{conn: conn}, nil
}

type tcpConnection struct {
	sync.Mutex
	conn net.Conn
}

func (c *tcpConnection) Send(msgName string, m Message, flags uint32) error {
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
	c.Lock()
	defer c.Unlock()
	var mh messageHeader
	if err := mh.ReadFromLike(c.conn, msgName); err != nil {
		return err
	}
	return m.ReadInto(c.conn)
}

func (c *tcpConnection) Close() error {
	return c.conn.Close()
}
