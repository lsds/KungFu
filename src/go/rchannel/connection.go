package rchannel

import (
	"log"
	"net"
	"sync"
)

// Connection encapsulates a TCP connection
type Connection struct {
	sendMu sync.Mutex
	recvMu sync.Mutex

	conn       net.Conn
	recvBuffer chan *Message
}

func newConnection(netAddr string, h connectionHeader) (*Connection, error) {
	conn, err := net.Dial("tcp", netAddr)
	if err != nil {
		return nil, err
	}
	if err := h.WriteTo(conn); err != nil {
		return nil, err
	}
	return &Connection{
		conn:       conn,
		recvBuffer: make(chan *Message, 10),
	}, nil
}

func (c *Connection) send(name string, m Message) error {
	log.Printf("%s::%s(%s, %s)", "Connection", "send", name, m)
	c.sendMu.Lock()
	defer c.sendMu.Unlock()
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

func (c *Connection) recv(name string, m *Message) error {
	c.recvMu.Lock()
	defer c.recvMu.Unlock()
	return nil
}

func (c *Connection) Close() error {
	return c.conn.Close()
}
