package kungfu

import (
	"fmt"
	"sync"
	"sync/atomic"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
)

// A strategy is a sequence of dataflow graph
type strategy struct {
	Graphs []*plan.Graph
}

// session contains the immutable topology and strategies for a given period of logical duration
type session struct {
	strategies []strategy
	cluster    *plan.ProcSpec
	router     *rch.Router
}

type partitionStrategy func([]plan.PeerSpec) []strategy

var partitionStrategies = map[kb.KungFu_AllReduceAlgo]partitionStrategy{
	kb.KungFu_Star:   createStarStrategies,
	kb.KungFu_Clique: createCliqueStrategies,
	kb.KungFu_Ring:   createRingStrategies,
	kb.KungFu_Tree:   createTreeStrategies,
}

func newSession(c Config, ps *plan.ProcSpec, router *rch.Router) *session {
	f := partitionStrategies[c.Algo]
	if f == nil {
		log.Warnf("%s is not implemeted, fallback to %s", c.Algo, kb.KungFu_Star)
		f = createStarStrategies
	}
	return &session{
		strategies: f(ps.Peers),
		cluster:    ps,
		router:     router,
	}
}

func createStarStrategies(peers []plan.PeerSpec) []strategy {
	k := len(peers)
	bcastGraph := plan.GenStarBcastGraph(k, 0)
	gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
	return []strategy{
		{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		},
	}
}

func createTreeStrategies(peers []plan.PeerSpec) []strategy {
	bcastGraph := plan.GenDefaultBcastGraph(peers)
	gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
	return []strategy{
		{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		},
	}
}

func createCliqueStrategies(peers []plan.PeerSpec) []strategy {
	k := len(peers)
	var ss []strategy
	for r := 0; r < k; r++ {
		bcastGraph := plan.GenStarBcastGraph(k, r)
		gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
		ss = append(ss, strategy{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		})
	}
	return ss
}

func createRingStrategies(peers []plan.PeerSpec) []strategy {
	k := len(peers)
	var ss []strategy
	for r := 0; r < k; r++ {
		gatherGraph, bcastGraph := plan.GenCircularGraphPair(k, r)
		ss = append(ss, strategy{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		})
	}
	return ss
}

func (sess *session) Warmup() int {
	k := sess.cluster.Size()
	count := k * 4
	dtype := kb.KungFu_INT32
	n := count * dtype.Size()
	w := Workspace{
		SendBuf: make([]byte, n),
		RecvBuf: make([]byte, n),
		Count:   count,
		Dtype:   dtype,
		OP:      kb.KungFu_SUM,
		Name:    "kungfu::warmup", // TODO: use tag
	}
	return code(sess.runStrategies(w, plan.EvenPartition, createCliqueStrategies(sess.cluster.Peers)))
}

func (sess *session) AllReduce(w Workspace) int {
	return code(sess.runStrategies(w, plan.EvenPartition, sess.strategies))
}

func (sess *session) runGraph(w Workspace, g *plan.Graph) error {
	sendTo := func(peer plan.PeerSpec) {
		sess.router.Send(peer.NetAddr.WithName(w.Name), w.RecvBuf)
	}

	var lock sync.Mutex
	recvAdd := func(peer plan.PeerSpec) {
		m := sess.router.Recv(peer.NetAddr.WithName(w.Name))
		lock.Lock()
		kb.Transform(w.RecvBuf, m.Data, w.Count, w.Dtype, w.OP)
		lock.Unlock()
	}

	recvAssign := func(peer plan.PeerSpec) {
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
	if g.IsSelfLoop(myRank) {
		copy(w.RecvBuf, w.SendBuf)
		prun(g.Prevs(myRank), recvAdd)
	} else {
		prevs := g.Prevs(myRank)
		if len(prevs) > 1 {
			log.Errorf("more than once recvAssign detected at node %d", myRank)
		}
		for _, rank := range prevs {
			recvAssign(sess.cluster.GetPeer(rank))
		}
	}
	prun(g.Nexts(myRank), sendTo)
	// TODO: handhel error
	return nil
}

func (sess *session) runGraphs(w Workspace, graphs ...*plan.Graph) error {
	if kc.InplaceAllReduce && len(graphs) == 2 { // FIXME: Assuming it is always a pair of allreduce graphs
		return sess.runAllReduceGraphPair(w, graphs[0], graphs[1])
	}
	for _, g := range graphs {
		// TODO: handhel error
		sess.runGraph(w, g)
	}
	return nil
}

func (sess *session) runStrategies(w Workspace, p partitionFunc, strategies []strategy) error {
	var wg sync.WaitGroup
	var failed int32
	for i, w := range w.split(p, len(strategies)) {
		wg.Add(1)
		go func(i int, w Workspace, s strategy) {
			if err := sess.runGraphs(w, s.Graphs...); err != nil {
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

func code(err error) int {
	if err == nil {
		return 0
	}
	// TODO: https://www.open-mpi.org/doc/v3.1/man3/MPI.3.php#sect4
	return 1
}
