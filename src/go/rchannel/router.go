package rchannel

import (
	"errors"
	"net"
	"strconv"
	"time"
)

type Router struct {
	localPort uint32
	buffers   map[Addr]chan *Message
	conns     map[string]*Connection
	chs       map[Addr]*Channel
}

var errNotFound = errors.New("Channel Not Found")

func NewRouter(cluster *ClusterSpec) (*Router, error) {
	port, err := strconv.Atoi(cluster.Self.Port)
	if err != nil {
		return nil, err
	}
	return &Router{
		localPort: uint32(port),
		buffers:   make(map[Addr]chan *Message), // in-comming messages
		conns:     make(map[string]*Connection), // out-going connections
		chs:       make(map[Addr]*Channel),
	}, nil
}

func (r *Router) getBuffer(a Addr) chan *Message {
	// TODO: mutex
	if _, ok := r.buffers[a]; !ok {
		r.buffers[a] = make(chan *Message, 10)
	}
	return r.buffers[a]
}

func (r *Router) getConnection(a Addr) (*Connection, error) {
	// TODO: mutex
	tk := time.NewTicker(100 * time.Millisecond)
	defer tk.Stop()
	netAddr := net.JoinHostPort(a.Host, a.Port)
	trials := 10
	for i := 0; i <= trials; i++ {
		if conn, ok := r.conns[netAddr]; !ok {
			conn, err := newConnection(netAddr, connectionHeader{Port: r.localPort})
			if err == nil {
				r.conns[netAddr] = conn
			}
		} else {
			return conn, nil
		}
		<-tk.C
	}
	return nil, errNotFound
}

// getChannel returns the Channel of given Addr
func (r *Router) getChannel(a Addr) (*Channel, error) {
	// TODO: mutex
	trials := 1
	for i := 0; i <= trials; i++ {
		if ch, ok := r.chs[a]; !ok {
			conn, err := r.getConnection(a)
			if err == nil {
				r.chs[a] = newChannel(a.Name, conn)
			}
		} else {
			return ch, nil
		}
	}
	return nil, errNotFound
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
	buffer := r.getBuffer(a)
	*m = *<-buffer
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
		buffer := r.getBuffer(remoteNetAddr.WithName(string(mh.Name)))
		buffer <- &m
	}
}
