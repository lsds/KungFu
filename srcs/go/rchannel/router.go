package rchannel

import (
	"errors"
	"fmt"
	"net"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/store"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Router struct {
	localAddr  plan.NetAddr
	Collective *CollectiveEndpoint // FIXME: move it out of Router
	connPool   *ConnectionPool
	monitor    monitor.Monitor

	store      *store.VersionedStore
	localStore *LocalStore // TODO: replaced by verison store

	reqMu sync.Mutex
}

func NewRouter(self plan.PeerID, store *store.VersionedStore) *Router {
	return &Router{
		localAddr:  plan.NetAddr(self),
		Collective: NewCollectiveEndpoint(),
		connPool:   newConnectionPool(), // out-going connections
		monitor:    monitor.GetMonitor(),
		store:      store,
		localStore: newLocalStore(), // TODO: replaced by verison store
	}
}

func (r *Router) Self() plan.PeerID {
	return plan.PeerID(r.localAddr)
}

// getChannel returns the Channel of given Addr
func (r *Router) getChannel(a plan.Addr, t ConnType) (*Channel, error) {
	conn, err := r.connPool.get(a.NetAddr(), r.localAddr, t)
	if err != nil {
		return nil, err
	}
	return newChannel(a.Name, conn), nil
}

func (r *Router) ResetConnections(keeps plan.PeerList) {
	r.connPool.reset(keeps)
}

// Request sends request name to given Addr
func (r *Router) Request(a plan.Addr, buf *kb.Vector) error {
	ch, err := r.getChannel(a, ConnPeerToPeer)
	if err != nil {
		return err
	}
	r.reqMu.Lock() // FIXME: lock per target
	defer r.reqMu.Unlock()
	if err := ch.Send(Message{}, NoFlag); err != nil {
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

func (r *Router) Pull(version string, a plan.Addr, buf *kb.Vector) error {
	ch, err := r.getChannel(a, ConnPeerToPeer)
	if err != nil {
		return err
	}
	r.reqMu.Lock() // FIXME: lock per target
	defer r.reqMu.Unlock()
	bs := []byte(version)
	if err := ch.Send(Message{Length: uint32(len(bs)), Data: bs}, NoFlag); err != nil {
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
func (r *Router) Send(a plan.Addr, buf []byte, t ConnType, flags uint32) error {
	msg := Message{
		Length: uint32(len(buf)),
		Data:   buf,
	}
	if err := r.send(a, msg, t, flags); err != nil {
		return err
	}
	r.monitor.Egress(int64(msg.Length), a.NetAddr())
	return nil
}

func (r *Router) send(a plan.Addr, msg Message, t ConnType, flags uint32) error {
	ch, err := r.getChannel(a, t)
	if err != nil {
		return err
	}
	if err := ch.Send(msg, flags); err != nil {
		return err
	}
	return nil
}

func (r *Router) handlePeerToPeerConn(name string, msg *Message, conn net.Conn, remote plan.NetAddr) {
	version := string(msg.Data)
	if len(version) == 0 {
		// TODO: Always using the verisoned tensor store.
		r.localStore.RLock()
		defer r.localStore.RUnlock()

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
	} else {
		// NOTE: This part is currently not used by any optimizer.
		var blob *store.Blob // FIXME: copy elision
		if err := r.store.Get(version, name, &blob); err != nil {
			utils.ExitErr(fmt.Errorf("Router.store.Get(%s, %s): %v", version, name, err))
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
			Length: uint32(len(blob.Data)),
			Data:   blob.Data,
		}
		if err := m.WriteTo(conn); err != nil {
			log.Errorf("Could not write variable from store to connection: %s", name)
			utils.ExitErr(err)
		}

		r.monitor.Egress(int64(m.Length), remote)
	}
}

func (r *Router) Save(name string, model *kb.Vector) error {
	r.localStore.Emplace(name, model)
	return nil
}

var ErrInvalidConnectionType = errors.New("invalid connection type")

// Handle implements ConnHandler.Handle interface
func (r *Router) Handle(conn net.Conn, remote plan.NetAddr, t ConnType) error {
	switch t {
	case ConnCollective:
		return r.Collective.Handle(conn, remote, t)
	case ConnPeerToPeer:
		_, err := Stream(conn, remote, Accept, r.handlePeerToPeerConn)
		return err
	default:
		return ErrInvalidConnectionType
	}
}
