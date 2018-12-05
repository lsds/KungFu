package rchannel

import (
	"net"
	"sync"
)

// Connection encapsulates a TCP connection
type Connection struct {
	sync.Mutex
	conn net.Conn
}

func newConnection(netAddr string, localPort uint32) (*Connection, error) {
	conn, err := net.Dial("tcp", netAddr)
	if err != nil {
		return nil, err
	}
	h := connectionHeader{Port: localPort}
	if err := h.WriteTo(conn); err != nil {
		return nil, err
	}
	return &Connection{
		conn: conn,
	}, nil
}

func (c *Connection) send(name string, m Message) error {
	// log.Printf("%s::%s(%s, %s)", "Connection", "send", name, m)
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

func (c *Connection) Close() error {
	return c.conn.Close()
}
