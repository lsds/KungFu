package session

import (
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/execution"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func (sess *Session) AllGather(w kb.Workspace) error {
	return sess.runAllGather(w)
}

func (sess *Session) runAllGather(w kb.Workspace) error {
	count := w.SendBuf.Count
	var sendInto execution.PeerFunc = func(peer plan.PeerID) error {
		return sess.client.Send(peer.WithName(w.Name), w.SendBuf.Data, connection.ConnCollective, connection.WaitRecvBuf)
	}
	var recvInto execution.PeerFunc = func(peer plan.PeerID) error {
		rank, ok := sess.peers.Rank(peer)
		if !ok {
			utils.Immpossible()
		}
		offset := rank * count
		sess.collectiveHandler.RecvInto(peer.WithName(w.Name), asMessage(w.RecvBuf.Slice(offset, offset+count)))
		return nil
	}
	others := sess.peers.Others(sess.self)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		sendInto.Par(others)
		wg.Done()
	}()
	go func() {
		recvInto.Par(others)
		wg.Done()
	}()
	w.RecvBuf.Slice(sess.rank*count, (sess.rank+1)*count).CopyFrom(w.SendBuf)
	wg.Wait()
	return nil // FIXME: handle errors
}
