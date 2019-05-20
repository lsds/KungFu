package kungfu

import (
	"errors"
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
	self       *plan.PeerSpec
	cluster    *plan.ClusterSpec
	myRank     int
	router     *rch.Router
}

type partitionStrategy func([]plan.PeerSpec) []strategy

var partitionStrategies = map[kb.KungFu_AllReduceAlgo]partitionStrategy{
	kb.KungFu_Star:   createStarStrategies,
	kb.KungFu_Clique: createCliqueStrategies,
	kb.KungFu_Ring:   createRingStrategies,
	kb.KungFu_Tree:   createTreeStrategies,
}

func newSession(c Config, self *plan.PeerSpec, cs *plan.ClusterSpec, router *rch.Router) (*session, error) {
	f := partitionStrategies[c.Algo]
	if f == nil {
		log.Warnf("%s is not implemeted, fallback to %s", c.Algo, kb.KungFu_Star)
		f = createStarStrategies
	}
	myRank, ok := cs.Lookup(*self)
	if !ok {
		return nil, errors.New("self not in cluster")
	}
	sess := &session{
		strategies: f(cs.Peers),
		self:       self,
		cluster:    cs,
		myRank:     myRank,
		router:     router,
	}
	if kc.RunWarmup {
		sess.Warmup() // TODO: check error
	}
	return sess, nil
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

func (sess *session) ClusterSize() int {
	return len(sess.cluster.Peers)
}

func (sess *session) Rank() int {
	return sess.myRank
}

func (sess *session) Warmup() int {
	k := len(sess.cluster.Peers)
	count := k * 4
	dtype := kb.KungFu_INT32
	w := Workspace{
		SendBuf: kb.NewBuffer(count, dtype),
		RecvBuf: kb.NewBuffer(count, dtype),
		OP:      kb.KungFu_SUM,
		Name:    "kungfu::warmup", // TODO: use tag
	}
	return code(sess.runStrategies(w, plan.EvenPartition, createCliqueStrategies(sess.cluster.Peers)))
}

func (sess *session) AllReduce(w Workspace) int {
	return code(sess.runStrategies(w, plan.EvenPartition, sess.strategies))
}

func (sess *session) Reduce(w Workspace) int {
	strategy := sess.strategies[0] // Assuming len(sess.strategies) > 0
	g := strategy.Graphs[0]        // Assuming the first graph is a Gather Graph
	return code(sess.runGraphs(w, g))
}

func (sess *session) Broadcast(w Workspace) int {
	strategy := sess.strategies[0] // Assuming len(sess.strategies) > 0
	g := strategy.Graphs[1]        // Assuming the second graph is a Broadcast Graph
	return code(sess.runGraphs(w, g))
}

func (sess *session) SendTo(rank int, w Workspace) int {
	log.Infof("session::SendTo(%d, w) from %d", rank, sess.myRank)
	if rank < 0 || len(sess.cluster.Peers) <= rank {
		return code(errInvalidRank)
	}
	peer := sess.cluster.Peers[rank]
	return code(sess.router.Send(peer.NetAddr.WithName(w.Name), w.SendBuf.Data, rch.ConnPeerToPeer))
}

func (sess *session) runGraphs(w Workspace, graphs ...*plan.Graph) error {
	if len(sess.cluster.Peers) == 1 {
		w.RecvBuf.CopyFrom(w.SendBuf)
		return nil
	}

	var recvCount int
	sendTo := func(peer plan.PeerSpec) {
		if recvCount == 0 {
			sess.router.Send(peer.NetAddr.WithName(w.Name), w.SendBuf.Data, rch.ConnCollective)
		} else {
			sess.router.Send(peer.NetAddr.WithName(w.Name), w.RecvBuf.Data, rch.ConnCollective)
		}
	}

	var lock sync.Mutex
	recvOnto := func(peer plan.PeerSpec) {
		m := sess.router.Recv(peer.NetAddr.WithName(w.Name))
		b := &kb.Buffer{Data: m.Data, Count: w.SendBuf.Count, Type: w.SendBuf.Type}
		lock.Lock()
		defer lock.Unlock()
		if recvCount == 0 {
			kb.Transform2(w.RecvBuf, w.SendBuf, b, w.OP)
		} else {
			kb.Transform(w.RecvBuf, b, w.OP)
		}
		recvCount++
	}

	recvInto := func(peer plan.PeerSpec) {
		m := sess.router.Recv(peer.NetAddr.WithName(w.Name))
		b := &kb.Buffer{Data: m.Data, Count: w.SendBuf.Count, Type: w.SendBuf.Type}
		w.RecvBuf.CopyFrom(b)
		recvCount++
	}

	par := func(ranks []int, op func(plan.PeerSpec)) {
		var wg sync.WaitGroup
		for _, rank := range ranks {
			wg.Add(1)
			go func(rank int) {
				op(sess.cluster.Peers[rank])
				wg.Done()
			}(rank)
		}
		wg.Wait()
	}

	seq := func(ranks []int, op func(plan.PeerSpec)) {
		for _, rank := range ranks {
			op(sess.cluster.Peers[rank])
		}
	}

	for _, g := range graphs {
		prevs := g.Prevs(sess.myRank)
		if g.IsSelfLoop(sess.myRank) {
			par(prevs, recvOnto)
		} else {
			if len(prevs) > 1 {
				log.Errorf("more than once recvInto detected at node %d", sess.myRank)
			}
			if len(prevs) == 0 && recvCount == 0 {
				w.RecvBuf.CopyFrom(w.SendBuf)
			} else {
				seq(prevs, recvInto) // len(prevs) == 1 is expected
			}
		}
		par(g.Nexts(sess.myRank), sendTo)
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

var (
	errInvalidRank = errors.New("invalid rank")
)

func code(err error) int {
	if err == nil {
		return 0
	}
	// TODO: https://www.open-mpi.org/doc/v3.1/man3/MPI.3.php#sect4
	return 1
}
