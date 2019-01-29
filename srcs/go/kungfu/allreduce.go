package kungfu

import (
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

// runAllReduceGraphPair is a specialization of runGraphs
func (sess *session) runAllReduceGraphPair(w Workspace, gather, bcast *plan.Graph) error {
	if sess.cluster.Size() == 1 {
		copy(w.RecvBuf, w.SendBuf)
		return nil
	}

	sendOriginTo := func(peer plan.PeerSpec) {
		sess.router.Send(peer.NetAddr.WithName(w.Name), w.SendBuf)
	}

	sendResultTo := func(peer plan.PeerSpec) {
		sess.router.Send(peer.NetAddr.WithName(w.Name), w.RecvBuf)
	}

	var lock sync.Mutex
	var gathered int
	recvOnto := func(peer plan.PeerSpec) {
		m := sess.router.Recv(peer.NetAddr.WithName(w.Name))
		lock.Lock()
		defer lock.Unlock()
		if gathered == 0 {
			kb.Transform2(w.RecvBuf, w.SendBuf, m.Data, w.Count, w.Dtype, w.OP)
		} else {
			kb.Transform(w.RecvBuf, m.Data, w.Count, w.Dtype, w.OP)
		}
		gathered++
	}

	recvInto := func(peer plan.PeerSpec) {
		m := sess.router.Recv(peer.NetAddr.WithName(w.Name))
		copy(w.RecvBuf, m.Data)
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
