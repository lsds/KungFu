package kungfu

import (
	"fmt"
	"sync"
	"sync/atomic"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

// Workspace contains the data that a Kungfu operation will be performed on.
type Workspace struct {
	SendBuf []byte
	RecvBuf []byte // TODO: if nil, will use SendBuf as in-place result
	Count   int
	Dtype   kb.KungFu_Datatype
	OP      kb.KungFu_Op
	Name    string
}

// 0 <= begin < end <= count - 1
func (w Workspace) slice(begin, end int) Workspace {
	i := begin * w.Dtype.Size()
	j := end * w.Dtype.Size()
	var recvBuf []byte
	if w.RecvBuf != nil {
		recvBuf = w.RecvBuf[i:j]
	}
	return Workspace{
		SendBuf: w.SendBuf[i:j],
		RecvBuf: recvBuf,
		Count:   end - begin,
		Dtype:   w.Dtype,
		OP:      w.OP,
		Name:    fmt.Sprintf("part::%s[%d:%d]", w.Name, begin, end),
	}
}

// partitionFunc is the signature of function that parts the interval
type partitionFunc func(r plan.Interval, k int) []plan.Interval

func (w Workspace) split(p partitionFunc, k int) []Workspace {
	var ws []Workspace
	for _, r := range p(plan.Interval{Begin: 0, End: w.Count}, k) {
		ws = append(ws, w.slice(r.Begin, r.End))
	}
	return ws
}

func (kf *Kungfu) runGraph(w Workspace, g *plan.Graph) error {
	sendTo := func(peer plan.PeerSpec) {
		kf.router.Send(peer.NetAddr.WithName(w.Name), w.RecvBuf)
	}

	var lock sync.Mutex
	recvAdd := func(peer plan.PeerSpec) {
		m := kf.router.Recv(peer.NetAddr.WithName(w.Name))
		lock.Lock()
		kb.Transform(w.RecvBuf, m.Data, w.Count, w.Dtype, w.OP)
		lock.Unlock()
	}

	recvAssign := func(peer plan.PeerSpec) {
		m := kf.router.Recv(peer.NetAddr.WithName(w.Name))
		copy(w.RecvBuf, m.Data)
	}

	cluster := kf.currentCluster()
	prun := func(ranks []int, op func(plan.PeerSpec)) {
		var wg sync.WaitGroup
		for _, rank := range ranks {
			wg.Add(1)
			go func(rank int) {
				op(cluster.GetPeer(rank))
				wg.Done()
			}(rank)
		}
		wg.Wait()
	}

	myRank := cluster.MyRank()
	if g.IsSelfLoop(myRank) {
		copy(w.RecvBuf, w.SendBuf)
		prun(g.Prevs(myRank), recvAdd)
	} else {
		prevs := g.Prevs(myRank)
		if len(prevs) > 1 {
			log.Errorf("more than once recvAssign detected at node %d", myRank)
		}
		for _, rank := range prevs {
			recvAssign(cluster.GetPeer(rank))
		}
	}
	prun(g.Nexts(myRank), sendTo)
	// TODO: handhel error
	return nil
}

func (kf *Kungfu) runGraphs(w Workspace, graphs ...*plan.Graph) error {
	if kc.InplaceAllReduce && len(graphs) == 2 { // FIXME: Assuming it is always a pair of allreduce graphs
		return kf.runAllReduceGraphPair(w, graphs[0], graphs[1])
	}
	for _, g := range graphs {
		// TODO: handhel error
		kf.runGraph(w, g)
	}
	return nil
}

func (kf *Kungfu) runStrategies(w Workspace, p partitionFunc, strategies []strategy) error {
	var wg sync.WaitGroup
	var failed int32
	for i, w := range w.split(p, len(strategies)) {
		wg.Add(1)
		go func(i int, w Workspace, s strategy) {
			if err := kf.runGraphs(w, s.Graphs...); err != nil {
				log.Warnf("partition %d failed: %v", i, err)
				atomic.AddInt32(&failed, 1)
			}
			wg.Done()
		}(i, w, strategies[i])
	}
	wg.Wait()
	if failed > 0 {
		return fmt.Errorf("%d strategies amoung %d failed", failed, len(strategies))
	}
	return nil
}
