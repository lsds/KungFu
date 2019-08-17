package rchannel

import (
	"fmt"
	"net"
	"os"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/shm"
	"github.com/lsds/KungFu/srcs/go/store"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Router struct {
	localAddr  plan.NetAddr
	bufferPool *BufferPool
	connPool   *ConnectionPool
	monitor    monitor.Monitor

	store      *store.VersionedStore
	localStore *LocalStore // FIXME: deprecated

	reqMu sync.Mutex
}

func NewRouter(self plan.PeerSpec, store *store.VersionedStore) *Router {
	return &Router{
		localAddr:  self.NetAddr,
		bufferPool: newBufferPool(),     // in-comming messages
		connPool:   newConnectionPool(), // out-going connections
		monitor:    monitor.GetMonitor(),
		store:      store,
		localStore: newLocalStore(), // FIXME: deprecated
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

// Request sends request name to given Addr
func (r *Router) Request(a plan.Addr, buf *kb.Buffer) error {
	ch, err := r.getChannel(a, ConnPeerToPeer)
	if err != nil {
		return err
	}
	r.reqMu.Lock() // FIXME: lock per target
	defer r.reqMu.Unlock()
	if err := ch.Send(Message{}); err != nil {
		return err
	}
	msg := Message{
		Length: uint32(buf.Count * buf.Type.Size()),
		Data:   buf.Data,
	}
	if err := ch.Receive(msg); err != nil {
		return err
	}
	r.monitor.Ingress(int64(msg.Length), a.NetAddr())
	return nil
}

func (r *Router) Pull(version string, a plan.Addr, buf *kb.Buffer) error {
	ch, err := r.getChannel(a, ConnPeerToPeer)
	if err != nil {
		return err
	}
	r.reqMu.Lock() // FIXME: lock per target
	defer r.reqMu.Unlock()
	bs := []byte(version)
	if err := ch.Send(Message{Length: uint32(len(bs)), Data: bs}); err != nil {
		return err
	}
	msg := Message{
		Length: uint32(buf.Count * buf.Type.Size()),
		Data:   buf.Data,
	}
	if err := ch.Receive(msg); err != nil {
		return err
	}
	r.monitor.Ingress(int64(msg.Length), a.NetAddr())
	return nil
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
	// log.Infof("%s::%s(%s)", "Router", "Recv", a)
	// TODO: reduce memory copy
	msg := *<-r.bufferPool.require(a)
	// TODO: add timeout
	return msg
}

func (r *Router) acceptOne(conn net.Conn, shm shm.Shm) (string, *Message, error) {
	var mh messageHeader
	if err := mh.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	var msg Message
	if mh.BodyInShm != 0 {
		log.Errorf("%s", "Should not get here")
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

func (r *Router) handlePeerToPeerConn(name string, msg *Message, conn net.Conn, remote plan.NetAddr) {
	version := string(msg.Data)
	if len(version) > 0 {
		var buf *kb.Buffer // FIXME: copy elision
		if err := r.store.Checkout(version, name, buf); err != nil {
			utils.ExitErr(err)
		}
		bs := []byte(name)
		mh := messageHeader{
			NameLength: uint32(len(bs)),
			Name:       bs,
		}
		if err := mh.WriteTo(conn); err != nil {
			log.Errorf("Could not write variable from store to connection: %s", name)
			utils.ExitErr(err)
		}
		m := Message{
			Length: uint32(buf.Count * buf.Type.Size()),
			Data:   buf.Data,
		}
		if err := m.WriteTo(conn); err != nil {
			log.Errorf("Could not write variable from store to connection: %s", name)
			utils.ExitErr(err)
		}
		r.monitor.Egress(int64(m.Length), remote)
		return
	}

	// FIXME: deprecated
	r.localStore.Lock()
	defer r.localStore.Unlock()

	modelBuffer := r.localStore.data[name]
	if modelBuffer == nil {
		utils.ExitErr(fmt.Errorf("Model buffer[%s] is nil", name))
	}

	bs := []byte(name)
	mh := messageHeader{
		NameLength: uint32(len(bs)),
		Name:       bs,
	}

	if err := mh.WriteTo(conn); err != nil {
		log.Errorf("Could not write variable from store to connection: %s", name)
		utils.ExitErr(err)
	}

	m := Message{
		Length: uint32(modelBuffer.Count * modelBuffer.Type.Size()),
		Data:   modelBuffer.Data,
	}

	if err := m.WriteTo(conn); err != nil {
		log.Errorf("Could not write variable from store to connection: %s", name)
		utils.ExitErr(err)
	}
	r.monitor.Egress(int64(m.Length), remote)
}

func (r *Router) Save(name string, model *kb.Buffer) error {
	r.localStore.Emplace(name, model)
	return nil
}

var newShm = shm.New

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
			r.handlePeerToPeerConn(name, msg, conn, remote)
		default:
			log.Infof("no handler for type %s", t)
		}
	}
}
