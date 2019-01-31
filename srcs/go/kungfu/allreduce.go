package kungfu

import (
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

// runAllReduceGraphPair is a specialization of runGraphs
func (sess *session) runAllReduceGraphPair(w Workspace, gather, bcast *plan.Graph) error {
	if sess.cluster.Size() == 1 {
		w.RecvBuf.CopyFrom(w.SendBuf)
		return nil
	}

	sendOriginTo := func(peer plan.PeerSpec) {
		sess.router.Send(peer.NetAddr.WithName(w.Name), w.SendBuf.Data)
	}

	sendResultTo := func(peer plan.PeerSpec) {
		sess.router.Send(peer.NetAddr.WithName(w.Name), w.RecvBuf.Data)
	}

	var lock sync.Mutex
	var gathered int
	recvOnto := func(peer plan.PeerSpec) {
		m := sess.router.Recv(peer.NetAddr.WithName(w.Name))
		b := &kb.Buffer{Data: m.Data, Count: w.SendBuf.Count, Type: w.SendBuf.Type}
		lock.Lock()
		defer lock.Unlock()
		if gathered == 0 {
			kb.Transform2(w.RecvBuf, w.SendBuf, b, w.OP)
		} else {
			kb.Transform(w.RecvBuf, b, w.OP)
		}
		gathered++
	}

	recvInto := func(peer plan.PeerSpec) {
		m := sess.router.Recv(peer.NetAddr.WithName(w.Name))
		b := &kb.Buffer{Data: m.Data, Count: w.SendBuf.Count, Type: w.SendBuf.Type}
		w.RecvBuf.CopyFrom(b)
	}

	prun := func(ranks []int, op func(plan.PeerSpec)) {
		var wg sync.WaitGroup
		for _, rank := range ranks {
			wg.Add(1)
			go func(rank int) {
				op(sess.cluster.GetPeer(rank))
				wg.Done()
			}(rank)
		}
		wg.Wait()
	}

	myRank := sess.cluster.MyRank()

	// stage1 :: Gather
	prun(gather.Prevs(myRank), recvOnto)
	// len(gather.Nexts(myRank)) <= 1
	if gathered == 0 {
		prun(gather.Nexts(myRank), sendOriginTo)
	} else {
		prun(gather.Nexts(myRank), sendResultTo)
	}

	// stage2 :: Bcast
	prun(bcast.Prevs(myRank), recvInto) // len(bcast.Prevs(myRank)) <= 1
	prun(bcast.Nexts(myRank), sendResultTo)

	return nil
}
