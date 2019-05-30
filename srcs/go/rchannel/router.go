package rchannel

import (
	"net"
	"os"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/shm"
)

type Callback func([]byte)

type Router struct {
	localAddr  plan.NetAddr
	bufferPool *BufferPool
	connPool   *ConnectionPool
	monitor    monitor.Monitor

	callbacks map[string]Callback // TODO: mutex
	// TODO: delele callbacks on exit
}

func NewRouter(self plan.PeerSpec) *Router {
	return &Router{
		localAddr:  self.NetAddr,
		bufferPool: newBufferPool(),     // in-comming messages
		connPool:   newConnectionPool(), // out-going connections
		monitor:    monitor.GetMonitor(),
		callbacks:  make(map[string]Callback),
	}
}

// getChannel returns the Channel of given Addr
func (r *Router) getChannel(a plan.Addr, t ConnType) (*Channel, error) {
	conn, err := r.connPool.get(a.NetAddr(), r.localAddr, t)
	if err != nil {
		return nil, err
	}
	return newChannel(a.Name, conn), nil
}

// Send sends data in buf to given Addr
func (r *Router) Send(a plan.Addr, buf []byte, t ConnType) error {
	msg := Message{
		Length: uint32(len(buf)),
		Data:   buf,
	}
	if err := r.send(a, msg, t); err != nil {
		log.Errorf("Router::Send failed: %v", err)
		// TODO: retry
		if t == ConnCollective {
			os.Exit(1)
		}
		// return err
	}
	r.monitor.Egress(int64(msg.Length), a.NetAddr())
	return nil
}

func (r *Router) send(a plan.Addr, msg Message, t ConnType) error {
	// log.Infof("%s::%s", "Router", "Send")
	ch, err := r.getChannel(a, t)
	if err != nil {
		return err
	}
	if err := ch.Send(msg); err != nil {
		return err
	}
	return nil
}

// Recv recevies a message from given Addr
func (r *Router) Recv(a plan.Addr) Message {
	log.Debugf("Router::Recv is waiting %s", a)
	// TODO: reduce memory copy
	msg := *<-r.bufferPool.require(a)
	// TODO: add timeout
	log.Debugf("Router::Recv got msg from %s", a)
	return msg
}

func (r *Router) acceptOne(conn net.Conn, shm shm.Shm) (string, *Message, error) {
	var mh messageHeader
	if err := mh.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	var msg Message
	if mh.BodyInShm != 0 {
		var mt messageTail
		if err := mt.ReadFrom(conn); err != nil {
			return "", nil, err
		}
		msg.Length = mt.Length
		msg.Data = make([]byte, msg.Length)
		shm.Seek(int(mt.Offset))
		shm.Read(msg.Data, int(msg.Length))
		mt.WriteTo(conn)
	} else {
		if err := msg.ReadFrom(conn); err != nil {
			return "", nil, err
		}
	}
	return string(mh.Name), &msg, nil
}

var newShm = shm.New

func (r *Router) RegisterDataCallback(name string, f Callback) {
	log.Infof("Router::RegisterDataCallback %s %p", name, f)
	// TODO: lock
	r.callbacks[name] = f
}

func (r *Router) UnregisterDataCallback(name string) {
	log.Infof("Router::UnregisterDataCallback %s", name)
	// TODO: lock
	delete(r.callbacks, name)
}

func (r *Router) handle(name string, msg *Message) {
	// TODO: lock
	f, ok := r.callbacks[name]
	if !ok {
		log.Warnf("%s has no callback registered", name)
		return
	}
	if f == nil {
		log.Errorf("%s has nil callback", name)
		return
	}
	// log.Infof("handling message with name %s", name)
	f(msg.Data)
}

func (r *Router) stream(conn net.Conn, remote plan.NetAddr, t ConnType) (int, error) {
	var shm shm.Shm
	if kc.UseShm && remote.Host == r.localAddr.Host {
		var err error
		if shm, err = newShm(plan.ShmNameFor(remote, r.localAddr)); err != nil {
			return 0, err
		}
		defer shm.Close()
	}
	for i := 0; ; i++ {
		name, msg, err := r.acceptOne(conn, shm)
		if err != nil {
			return i, err
		}
		r.monitor.Ingress(int64(msg.Length), remote)
		switch t {
		case ConnCollective:
			r.bufferPool.require(remote.WithName(name)) <- msg
		case ConnPeerToPeer:
			r.handle(name, msg)
		default:
			log.Infof("no handler for type %s", t)
		}
	}
}
