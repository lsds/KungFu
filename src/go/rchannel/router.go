package rchannel

import (
	"net"
	"strconv"
)

type Router struct {
	localPort  uint32
	bufferPool *BufferPool
	connPool   *ConnectionPool
}

func NewRouter(cluster *ClusterSpec) (*Router, error) {
	port, err := strconv.Atoi(cluster.Self.Port)
	if err != nil {
		return nil, err
	}
	return &Router{
		localPort:  uint32(port),
		bufferPool: newBufferPool(),     // in-comming messages
		connPool:   newConnectionPool(), // out-going connections
	}, nil
}

// getChannel returns the Channel of given Addr
func (r *Router) getChannel(a Addr) (*Channel, error) {
	netAddr := net.JoinHostPort(a.Host, a.Port)
	conn, err := r.connPool.get(netAddr, r.localPort)
	if err != nil {
		return nil, err
	}
	return newChannel(a.Name, conn), nil
}

// Send sends Message to given Addr
func (r *Router) Send(a Addr, m Message) error {
	// log.Printf("%s::%s", "Router", "Send")
	ch, err := r.getChannel(a)
	if err != nil {
		return err
	}
	return ch.Send(m)
}

// Recv recevies a message from given Addr
func (r *Router) Recv(a Addr, m *Message) error {
	// log.Printf("%s::%s(%s)", "Router", "Recv", a)
	// TODO: reduce memory copy
	*m = *<-r.bufferPool.require(a)
	// TODO: add timeout
	return nil
}

func (r *Router) stream(conn net.Conn, remoteNetAddr NetAddr) error {
	for {
		var mh messageHeader
		if err := mh.ReadFrom(conn); err != nil {
			return err
		}
		// log.Printf("got message header: %s", mh)
		var m Message
		if err := m.ReadFrom(conn); err != nil {
			return err
		}
		// log.Printf("got message: %s :: %s", m, string(m.Data))
		a := remoteNetAddr.WithName(string(mh.Name))
		r.bufferPool.require(a) <- &m
	}
}
