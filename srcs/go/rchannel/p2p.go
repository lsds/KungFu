package rchannel

import (
	"net"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/store"
)

type PeerToPeerEndpoint struct {
	store      *store.VersionedStore
	localStore *LocalStore // TODO: replaced by verison store

	waitQ *BufferPool
	recvQ *BufferPool

	router *Router
}

func NewPeerToPeerEndpoint(router *Router) *PeerToPeerEndpoint {
	return &PeerToPeerEndpoint{
		store:      store.NewVersionedStore(3),
		localStore: newLocalStore(), // TODO: replaced by verison store
		waitQ:      newBufferPool(1),
		recvQ:      newBufferPool(1),
		router:     router,
	}
}

// Handle implements ConnHandler.Handle interface
func (e *PeerToPeerEndpoint) Handle(conn net.Conn, remote plan.NetAddr, t ConnType) error {
	if t != ConnPeerToPeer {
		return ErrInvalidConnectionType
	}
	_, err := Stream(conn, remote, e.accept, e.handle)
	return err
}

func (e *PeerToPeerEndpoint) RecvInto(a plan.Addr, m Message) (bool, error) {
	e.waitQ.require(a) <- &m
	pm := <-e.recvQ.require(a)
	if !m.same(pm) {
		return false, errRegisteredBufferNotUsed
	}
	return !pm.hasFlag(RequestFailed), nil
}

func (e *PeerToPeerEndpoint) Save(name string, model *kb.Vector) error {
	e.localStore.Emplace(name, model)
	return nil
}

func (e *PeerToPeerEndpoint) SaveVersion(version, name string, buf *kb.Vector) error {
	blob := &store.Blob{Data: buf.Data}
	return e.store.Create(version, name, blob)
}

func (e *PeerToPeerEndpoint) accept(conn net.Conn, remote plan.NetAddr) (string, *Message, error) {
	var mh messageHeader
	if err := mh.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	name := string(mh.Name)
	if mh.HasFlag(IsResponse) {
		m := <-e.waitQ.require(remote.WithName(name))
		m.flags = mh.Flags
		if mh.HasFlag(RequestFailed) {
			return name, m, nil
		}
		if err := m.ReadInto(conn); err != nil {
			return "", nil, err
		}
		return name, m, nil
	}
	var m Message
	m.flags = mh.Flags
	if err := m.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	return name, &m, nil
}

func (e *PeerToPeerEndpoint) handle(name string, msg *Message, conn net.Conn, remote plan.NetAddr) {
	if msg.hasFlag(IsResponse) {
		e.recvQ.require(remote.WithName(name)) <- msg
		return
	}
	e.response(name, string(msg.Data), remote)
}

func (e *PeerToPeerEndpoint) response(name, version string, remote plan.NetAddr) error {
	ch, err := e.router.getChannel(remote.WithName(name), ConnPeerToPeer)
	if err != nil {
		return err
	}
	flags := IsResponse
	var buf []byte
	if len(version) == 0 {
		e.localStore.RLock()
		defer e.localStore.RUnlock()
		if v, ok := e.localStore.data[name]; ok {
			buf = v.Data
		} else {
			flags |= RequestFailed
		}
	} else {
		var blob *store.Blob // FIXME: copy elision
		if err := e.store.Get(version, name, &blob); err == nil {
			buf = blob.Data
		} else {
			flags |= RequestFailed
		}
	}
	msg := Message{
		Length: uint32(len(buf)),
		Data:   buf,
	}
	return ch.Send(msg, flags)
}
