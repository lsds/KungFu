package kungfu

import (
	"errors"
	"fmt"
	"sync"

	"github.com/lsds/KungFu/srcs/go/utils"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
)

const defaultRoot = 0

// A strategy is a pair of dataflow graphs
type strategy struct {
	reduceGraph *plan.Graph
	bcastGraph  *plan.Graph
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

func newSession(c Config, self *plan.PeerSpec, cs *plan.ClusterSpec, router *rch.Router) (*session, bool, error) {
	f := partitionStrategies[c.Algo]
	if f == nil {
		log.Warnf("%s is not implemeted, fallback to %s", c.Algo, kb.KungFu_Star)
		f = createStarStrategies
	}
	myRank, ok := cs.Lookup(*self)
	if !ok {
		return nil, false, nil
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
	return sess, true, nil
}

func createStarStrategies(peers []plan.PeerSpec) []strategy {
	k := len(peers)
	bcastGraph := plan.GenStarBcastGraph(k, defaultRoot)
	reduceGraph := plan.GenDefaultReduceGraph(bcastGraph)
	return []strategy{
		{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		},
	}
}

func createTreeStrategies(peers []plan.PeerSpec) []strategy {
	bcastGraph := plan.GenDefaultBcastGraph(peers)
	reduceGraph := plan.GenDefaultReduceGraph(bcastGraph)
	return []strategy{
		{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		},
	}
}

func createCliqueStrategies(peers []plan.PeerSpec) []strategy {
	k := len(peers)
	var ss []strategy
	for r := 0; r < k; r++ {
		bcastGraph := plan.GenStarBcastGraph(k, r)
		reduceGraph := plan.GenDefaultReduceGraph(bcastGraph)
		ss = append(ss, strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		})
	}
	return ss
}

func createRingStrategies(peers []plan.PeerSpec) []strategy {
	k := len(peers)
	var ss []strategy
	for r := 0; r < k; r++ {
		reduceGraph, bcastGraph := plan.GenCircularGraphPair(k, r)
		ss = append(ss, strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
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

func (sess *session) Barrier() int {
	k := len(sess.cluster.Peers)
	count := k * 1
	dtype := kb.KungFu_UINT8
	w := Workspace{
		SendBuf: kb.NewBuffer(count, dtype),
		RecvBuf: kb.NewBuffer(count, dtype),
		OP:      kb.KungFu_SUM,
		Name:    "kungfu::barrier", // TODO: use tag
	}
	return code(sess.runStrategies(w, plan.EvenPartition, createCliqueStrategies(sess.cluster.Peers)))
}

func (sess *session) AllReduce(w Workspace) int {
	return code(sess.runStrategies(w, plan.EvenPartition, sess.strategies))
}

func (sess *session) Reduce(w Workspace) int {
	strategy := sess.strategies[0] // Assuming len(sess.strategies) > 0
	return code(sess.runGraphs(w, strategy.reduceGraph))
}

func (sess *session) Broadcast(w Workspace) int {
	strategy := sess.strategies[0] // Assuming len(sess.strategies) > 0
	return code(sess.runGraphs(w, strategy.bcastGraph))
}

func (sess *session) Gather(w Workspace) int {
	// TODO: validate input
	return code(sess.runGather(w))
}

func (sess *session) Request(rank int, name string, model *kb.Buffer) int {
	if rank < 0 || len(sess.cluster.Peers) <= rank {
		return code(errInvalidRank)
	}
	peer := sess.cluster.Peers[rank]
	return code(sess.router.Request(peer.NetAddr.WithName(name), model))
}

func (sess *session) Pull(rank int, version, name string, model *kb.Buffer) int {
	peer := sess.cluster.Peers[rank]
	return code(sess.router.Pull(version, peer.NetAddr.WithName(name), model))
}

// FIXME: move it to kungfu
func (sess *session) Save(name string, buf *kb.Buffer) int {
	return code(sess.router.Save(name, buf))
}

func asMessage(b *kb.Buffer) rch.Message {
	return rch.Message{
		Length: uint32(len(b.Data)),
		Data:   b.Data,
	}
}

func (sess *session) runGather(w Workspace) error {
	if sess.myRank != defaultRoot {
		peer := sess.cluster.Peers[defaultRoot]
		return sess.router.Send(peer.NetAddr.WithName(w.Name), w.SendBuf.Data, rch.ConnCollective, rch.NoFlag)
	}
	var wg sync.WaitGroup
	count := w.SendBuf.Count
	for rank, peer := range sess.cluster.Peers {
		wg.Add(1)
		go func(rank int, peer plan.PeerSpec, recvBuf *kb.Buffer) {
			if rank == sess.myRank {
				recvBuf.CopyFrom(w.SendBuf)
			} else {
				m := sess.router.Recv(peer.NetAddr.WithName(w.Name))
				b := &kb.Buffer{Data: m.Data, Count: recvBuf.Count, Type: recvBuf.Type}
				recvBuf.CopyFrom(b)

				// Return m
			}
			wg.Done()
		}(rank, peer, w.RecvBuf.Slice(count*rank, count*(rank+1)))
	}
	wg.Wait()
	return nil // FIXME: handle errors
}

func (sess *session) runGraphs(w Workspace, graphs ...*plan.Graph) error {
	if len(sess.cluster.Peers) == 1 {
		w.RecvBuf.CopyFrom(w.SendBuf)
		return nil
	}

	var recvCount int
	effectiveData := func() []byte {
		if recvCount == 0 {
			return w.SendBuf.Data
		}
		return w.RecvBuf.Data
	}
	sendOnto := func(peer plan.PeerSpec) error {
		return sess.router.Send(peer.NetAddr.WithName(w.Name), effectiveData(), rch.ConnCollective, rch.NoFlag)
	}
	sendInto := func(peer plan.PeerSpec) error {
		return sess.router.Send(peer.NetAddr.WithName(w.Name), effectiveData(), rch.ConnCollective, rch.WaitRecvBuf)
	}

	var lock sync.Mutex
	recvOnto := func(peer plan.PeerSpec) error {
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
		
		// Return m
		return nil
	}

	recvInto := func(peer plan.PeerSpec) {
		sess.router.RecvInto(peer.NetAddr.WithName(w.Name), asMessage(w.RecvBuf))
		recvCount++
	}

	par := func(ranks []int, op func(plan.PeerSpec) error) error {
		errs := make([]error, len(ranks))
		var wg sync.WaitGroup
		for i, rank := range ranks {
			wg.Add(1)
			go func(i, rank int) {
				errs[i] = op(sess.cluster.Peers[rank])
				wg.Done()
			}(i, rank)
		}
		wg.Wait()
		return mergeErrors(errs, "par")
	}

	seq := func(ranks []int, op func(plan.PeerSpec)) {
		for _, rank := range ranks {
			op(sess.cluster.Peers[rank])
		}
	}

	for _, g := range graphs {
		prevs := g.Prevs(sess.myRank)
		if g.IsSelfLoop(sess.myRank) {
			if err := par(prevs, recvOnto); err != nil {
				return err
			}
			if err := par(g.Nexts(sess.myRank), sendOnto); err != nil {
				return err
			}
		} else {
			if len(prevs) > 1 {
				log.Errorf("more than once recvInto detected at node %d", sess.myRank)
			}
			if len(prevs) == 0 && recvCount == 0 {
				w.RecvBuf.CopyFrom(w.SendBuf)
			} else {
				seq(prevs, recvInto) // len(prevs) == 1 is expected
			}
			if err := par(g.Nexts(sess.myRank), sendInto); err != nil {
				return err
			}
		}
	}
	return nil
}

const (
	Mi        = 1 << 20
	chunkSize = 1 * Mi
)

func ceilDiv(a, b int) int {
	if a%b == 0 {
		return a / b
	}
	return a/b + 1
}

func (sess *session) runStrategies(w Workspace, p partitionFunc, strategies []strategy) error {
	k := ceilDiv(w.RecvBuf.Count*w.RecvBuf.Type.Size(), chunkSize)
	errs := make([]error, k)
	var wg sync.WaitGroup
	for i, w := range w.split(p, k) {
		wg.Add(1)
		go func(i int, w Workspace, s strategy) {
			errs[i] = sess.runGraphs(w, s.reduceGraph, s.bcastGraph)
			wg.Done()
		}(i, w, strategies[i%len(strategies)])
	}
	wg.Wait()
	return mergeErrors(errs, "runStrategies")
}

var (
	errInvalidRank = errors.New("invalid rank")
)

func code(err error) int {
	if err == nil {
		return 0
	}
	log.Errorf("kungfu operation failed: %v", err)
	// TODO: https://www.open-mpi.org/doc/v3.1/man3/MPI.3.php#sect4
	return 1
}

func mergeErrors(errs []error, hint string) error {
	var msg string
	var failed int
	for _, e := range errs {
		if e != nil {
			failed++
			if len(msg) > 0 {
				msg += ", "
			}
			msg += e.Error()
		}
	}
	if failed == 0 {
		return nil
	}
	return fmt.Errorf("%s failed with %d %s: %s", hint, failed, utils.Pluralize(failed, "error", "errors"), msg)
}
