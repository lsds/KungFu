package handler

import (
	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/client"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/store"
)

const defaultVersionCount = 3

type PeerToPeerEndpoint struct {
	versionedStore *store.VersionedStore
	store          *store.Store
	waitQ          *BufferPool
	recvQ          *BufferPool
	client         *client.Client
}

func NewPeerToPeerEndpoint(client *client.Client) *PeerToPeerEndpoint {
	return &PeerToPeerEndpoint{
		versionedStore: store.NewVersionedStore(defaultVersionCount),
		store:          store.NewStore(),
		waitQ:          newBufferPool(1),
		recvQ:          newBufferPool(1),
		client:         client,
	}
}

// Handle implements ConnHandler.Handle interface
func (e *PeerToPeerEndpoint) Handle(conn connection.Connection) (int, error) {
	return connection.Stream(conn, e.accept, e.handle)
}

func (e *PeerToPeerEndpoint) Request(a plan.Addr, version string, m connection.Message) (bool, error) {
	e.waitQ.require(a) <- &m
	if err := e.client.Send(a, []byte(version), connection.ConnPeerToPeer, connection.NoFlag); err != nil {
		<-e.waitQ.require(a)
		return false, err // FIXME: allow send to fail
	}
	pm := <-e.recvQ.require(a)
	if !m.Same(pm) {
		return false, errRegisteredBufferNotUsed
	}
	return !pm.HasFlag(connection.RequestFailed), nil
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

func (e *PeerToPeerEndpoint) accept(conn connection.Connection) (string, *connection.Message, error) {
	var mh connection.MessageHeader
	if err := mh.ReadFrom(conn.Conn()); err != nil {
		return "", nil, err
	}
	name := string(mh.Name)
	if mh.HasFlag(connection.IsResponse) {
		m := <-e.waitQ.require(conn.Src().WithName(name))
		m.Flags = mh.Flags
		if mh.HasFlag(connection.RequestFailed) {
			var empty connection.Message
			if err := empty.ReadInto(conn.Conn()); err != nil {
				return "", nil, err
			}
			return name, m, nil
		}
		if err := m.ReadInto(conn.Conn()); err != nil {
			return "", nil, err
		}
		return name, m, nil
	}
	var m connection.Message
	m.Flags = mh.Flags
	if err := m.ReadFrom(conn.Conn()); err != nil {
		return "", nil, err
	}
	return name, &m, nil
}

func (e *PeerToPeerEndpoint) handle(name string, msg *connection.Message, conn connection.Connection) {
	if msg.HasFlag(connection.IsResponse) {
		e.recvQ.require(conn.Src().WithName(name)) <- msg
		return
	}
	go e.response(name, msg.Data, conn.Src()) // FIXME: check error, use one queue
}

func (e *PeerToPeerEndpoint) response(name string, version []byte, remote plan.PeerID) error {
	var blob *store.Blob
	var err error
	if len(version) == 0 {
		blob, err = e.store.Get(name)
	} else {
		blob, err = e.versionedStore.Get(string(version), name)
	}
	flags := connection.IsResponse
	var buf []byte
	if err == nil {
		blob.RLock()
		defer blob.RUnlock()
		buf = blob.Data
	} else {
		flags |= connection.RequestFailed
	}
	return e.client.Send(remote.WithName(name), buf, connection.ConnPeerToPeer, flags)
}
