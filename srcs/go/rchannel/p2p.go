package rchannel

import (
	"fmt"
	"net"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/store"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type PeerToPeerEndpoint struct {
	store      *store.VersionedStore
	localStore *LocalStore // TODO: replaced by verison store
}

func NewPeerToPeerEndpoint() *PeerToPeerEndpoint {
	return &PeerToPeerEndpoint{
		store:      store.NewVersionedStore(3),
		localStore: newLocalStore(), // TODO: replaced by verison store
	}
}

// Handle implements ConnHandler.Handle interface
func (e *PeerToPeerEndpoint) Handle(conn net.Conn, remote plan.NetAddr, t ConnType) error {
	if t != ConnPeerToPeer {
		return ErrInvalidConnectionType
	}
	_, err := Stream(conn, remote, Accept, e.handle)
	return err
}

func (e *PeerToPeerEndpoint) Save(name string, model *kb.Vector) error {
	e.localStore.Emplace(name, model)
	return nil
}

func (e *PeerToPeerEndpoint) SaveVersion(version, name string, buf *kb.Vector) error {
	blob := &store.Blob{Data: buf.Data}
	return e.store.Create(version, name, blob)
}

func (e *PeerToPeerEndpoint) handle(name string, msg *Message, conn net.Conn, remote plan.NetAddr) {
	version := string(msg.Data)
	if len(version) == 0 {
		// TODO: Always using the verisoned tensor store.
		e.localStore.RLock()
		defer e.localStore.RUnlock()

		modelBuffer := e.localStore.data[name]
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

		// r.monitor.Egress(int64(m.Length), remote)
	} else {
		// NOTE: This part is currently not used by any optimizer.
		var blob *store.Blob // FIXME: copy elision
		if err := e.store.Get(version, name, &blob); err != nil {
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

		// r.monitor.Egress(int64(m.Length), remote)
	}
}
