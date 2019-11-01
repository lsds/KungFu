package rchannel

import (
	"net"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/store"
)

const defaultVersionCount = 3

type PeerToPeerEndpoint struct {
	versionedStore *store.VersionedStore
	store          *store.Store
	waitQ          *BufferPool
	recvQ          *BufferPool
	router         *Router
}

func NewPeerToPeerEndpoint(router *Router) *PeerToPeerEndpoint {
	return &PeerToPeerEndpoint{
		versionedStore: store.NewVersionedStore(defaultVersionCount),
		store:          store.NewStore(),
		waitQ:          newBufferPool(1),
		recvQ:          newBufferPool(1),
		router:         router,
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

func (e *PeerToPeerEndpoint) Request(a plan.Addr, version string, m Message) (bool, error) {
	e.waitQ.require(a) <- &m
	if err := e.router.Send(a, []byte(version), ConnPeerToPeer, NoFlag); err != nil {
		<-e.waitQ.require(a)
		return false, err // FIXME: allow send to fail
	}
	pm := <-e.recvQ.require(a)
	if !m.same(pm) {
		return false, errRegisteredBufferNotUsed
	}
	return !pm.hasFlag(RequestFailed), nil
}

func (e *PeerToPeerEndpoint) Save(name string, buf *kb.Vector) error {
	blob, err := e.store.GetOrCreate(name, len(buf.Data))
	if err != nil {
		return err
	}
	return blob.CopyFrom(buf.Data)
}

func (e *PeerToPeerEndpoint) SaveVersion(version, name string, buf *kb.Vector) error {
	blob, err := e.versionedStore.GetOrCreate(version, name, len(buf.Data))
	if err != nil {
		return err
	}
	return blob.CopyFrom(buf.Data)
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
			var empty Message
			if err := empty.ReadInto(conn); err != nil {
				return "", nil, err
			}
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
	go e.response(name, msg.Data, remote) // FIXME: check error, use one queue
}

func (e *PeerToPeerEndpoint) response(name string, version []byte, remote plan.NetAddr) error {
	var blob *store.Blob
	var err error
	if len(version) == 0 {
		blob, err = e.store.Get(name)
	} else {
		blob, err = e.versionedStore.Get(string(version), name)
	}
	flags := IsResponse
	var buf []byte
	if err == nil {
		blob.RLock()
		defer blob.RUnlock()
		buf = blob.Data
	} else {
		flags |= RequestFailed
	}
	return e.router.Send(remote.WithName(name), buf, ConnPeerToPeer, flags)
}
