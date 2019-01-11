package kungfu

import (
	"fmt"
	"sync"
	"sync/atomic"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
)

func (kf *Kungfu) runGraph(sendBuf []byte, recvBuf []byte, count int, dtype kb.KungFu_Datatype, op kb.KungFu_Op, name string, g *plan.Graph) error {
	n := count * dtype.Size()

	sendTo := func(rank int) {
		task := kf.currentCluster().GetPeer(rank)
		m := rch.NewMessage(recvBuf[:n])
		kf.router.Send(task.NetAddr.WithName(name), *m)
	}

	var lock sync.Mutex
	recvAdd := func(rank int) {
		task := kf.currentCluster().GetPeer(rank)
		addr := task.NetAddr
		var m rch.Message
		kf.router.Recv(addr.WithName(name), &m)
		lock.Lock()
		kb.Transform(recvBuf[:n], m.Data, count, dtype, op)
		lock.Unlock()
	}

	recvAssign := func(rank int) {
		task := kf.currentCluster().GetPeer(rank)
		addr := task.NetAddr
		var m rch.Message
		kf.router.Recv(addr.WithName(name), &m)
		copy(recvBuf[:n], m.Data)
	}

	prun := func(ranks []int, op func(int)) {
		var wg sync.WaitGroup
		for _, rank := range ranks {
			wg.Add(1)
			go func(rank int) {
				op(rank)
				wg.Done()
			}(rank)
		}
		wg.Wait()
	}

	myRank := kf.currentCluster().MyRank()
	if g.IsSelfLoop(myRank) {
		copy(recvBuf[:n], sendBuf[:n])
		prun(g.Prevs(myRank), recvAdd)
	} else {
		prevs := g.Prevs(myRank)
		if len(prevs) > 1 {
			log.Errorf("more than once recvAssign detected at node %d", myRank)
		}
		for _, prev := range prevs {
			recvAssign(prev)
		}
	}
	prun(g.Nexts(myRank), sendTo)
	// TODO: handhel error
	return nil
}

func (kf *Kungfu) runGraphs(sendBuf []byte, recvBuf []byte, count int, dtype kb.KungFu_Datatype, op kb.KungFu_Op, name string, graphs ...*plan.Graph) error {
	for _, g := range graphs {
		// TODO: handhel error
		kf.runGraph(sendBuf, recvBuf, count, dtype, op, name, g)
	}
	return nil
}

// partitionFunc is the signature of function that parts the interval
type partitionFunc func(r plan.Interval, k int) []plan.Interval

func (kf *Kungfu) runPartitions(sendBuf []byte, recvBuf []byte, count int, dtype kb.KungFu_Datatype, op kb.KungFu_Op, name string, split partitionFunc, partitions []partition) error {
	ranges := split(plan.Interval{Begin: 0, End: count}, len(partitions))
	var wg sync.WaitGroup
	var failed int32
	for i, p := range partitions {
		wg.Add(1)
		go func(i int, r plan.Interval, p partition) {
			x := r.Begin * dtype.Size()
			y := r.End * dtype.Size()
			fullName := fmt.Sprintf("part::%s[%d:%d]", name, r.Begin, r.End) // TODO: use tag
			if err := kf.runGraphs(sendBuf[x:y], recvBuf[x:y], r.Len(), dtype, op, fullName, p.Graphs...); err != nil {
				log.Warnf("partition %d failed: %v", i, err)
				atomic.AddInt32(&failed, 1)
			}
			wg.Done()
		}(i, ranges[i], p)
	}
	wg.Wait()
	if failed > 0 {
		return fmt.Errorf("%d partitions amoung %d failed", failed, len(partitions))
	}
	return nil
}
