package kungfu

import (
	"errors"
	"fmt"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	"github.com/lsds/KungFu/srcs/go/utils"
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
	self       plan.PeerID
	peers      plan.PeerList
	myRank     int
	router     *rch.Router
}

func newSession(strategy kb.Strategy, self plan.PeerID, pl plan.PeerList, router *rch.Router) (*session, bool) {
	myRank, ok := pl.Lookup(self)
	if !ok {
		return nil, false
	}
	if strategy == kb.Auto {
		strategy = autoSelect(pl)
	}
	sess := &session{
		strategies: partitionStrategies[strategy](pl),
		self:       self,
		peers:      pl,
		myRank:     myRank,
		router:     router,
	}
	return sess, true
}

func (sess *session) ClusterSize() int {
	return len(sess.peers)
}

func (sess *session) Rank() int {
	return sess.myRank
}

func (sess *session) Barrier() error {
	return sess.barrier()
}

func (sess *session) barrier() error {
	k := len(sess.peers)
	count := k * 1
	dtype := kb.U8
	w := Workspace{
		SendBuf: kb.NewVector(count, dtype),
		RecvBuf: kb.NewVector(count, dtype),
		OP:      kb.SUM,
		Name:    "kungfu::barrier", // TODO: use tag
	}
	return sess.runStrategies(w, plan.EvenPartition, sess.strategies)
}

func (sess *session) AllReduce(w Workspace) error {
	return sess.runStrategies(w, plan.EvenPartition, sess.strategies)
}

func (sess *session) Reduce(w Workspace) error {
	strategy := sess.strategies[0] // Assuming len(sess.strategies) > 0
	return sess.runGraphs(w, strategy.reduceGraph)
}

func (sess *session) Broadcast(w Workspace) error {
	strategy := sess.strategies[0] // Assuming len(sess.strategies) > 0
	return sess.runGraphs(w, strategy.bcastGraph)
}

func (sess *session) Gather(w Workspace) error {
	// TODO: validate input
	return sess.runGather(w)
}

func (sess *session) Request(rank int, name string, model *kb.Vector) error {
	if rank < 0 || len(sess.peers) <= rank {
		return errInvalidRank
	}
	peer := sess.peers[rank]
	return sess.router.Request(peer.WithName(name), model)
}

func (sess *session) Pull(rank int, version, name string, model *kb.Vector) error {
	peer := sess.peers[rank]
	return sess.router.Pull(version, peer.WithName(name), model)
}

// FIXME: move it to kungfu
func (sess *session) Save(name string, buf *kb.Vector) error {
	return sess.router.Save(name, buf)
}

func asMessage(b *kb.Vector) rch.Message {
	return rch.Message{
		Length: uint32(len(b.Data)),
		Data:   b.Data,
	}
}

func (sess *session) runGather(w Workspace) error {
	if sess.myRank != defaultRoot {
		peer := sess.peers[defaultRoot]
		return sess.router.Send(peer.WithName(w.Name), w.SendBuf.Data, rch.ConnCollective, rch.NoFlag)
	}
	var wg sync.WaitGroup
	count := w.SendBuf.Count
	for rank, peer := range sess.peers {
		wg.Add(1)
		go func(rank int, peer plan.PeerID, recvBuf *kb.Vector) {
			if rank == sess.myRank {
				recvBuf.CopyFrom(w.SendBuf)
			} else {
				m := sess.router.Recv(peer.WithName(w.Name))
				b := &kb.Vector{Data: m.Data, Count: recvBuf.Count, Type: recvBuf.Type}
				recvBuf.CopyFrom(b)
			}
			wg.Done()
		}(rank, peer, w.RecvBuf.Slice(count*rank, count*(rank+1)))
	}
	wg.Wait()
	return nil // FIXME: handle errors
}

func (sess *session) runGraphs(w Workspace, graphs ...*plan.Graph) error {
	if len(sess.peers) == 1 {
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
	sendOnto := func(peer plan.PeerID) error {
		return sess.router.Send(peer.WithName(w.Name), effectiveData(), rch.ConnCollective, rch.NoFlag)
	}
	sendInto := func(peer plan.PeerID) error {
		return sess.router.Send(peer.WithName(w.Name), effectiveData(), rch.ConnCollective, rch.WaitRecvBuf)
	}

	var lock sync.Mutex
	recvOnto := func(peer plan.PeerID) error {
		m := sess.router.Recv(peer.WithName(w.Name))
		b := &kb.Vector{Data: m.Data, Count: w.SendBuf.Count, Type: w.SendBuf.Type}
		lock.Lock()
		defer lock.Unlock()
		if recvCount == 0 {
			kb.Transform2(w.RecvBuf, w.SendBuf, b, w.OP)
		} else {
			kb.Transform(w.RecvBuf, b, w.OP)
		}
		recvCount++
		rch.PutBuf(m.Data) // Recycle buffer on the RecvOnto path
		return nil
	}

	recvInto := func(peer plan.PeerID) {
		sess.router.RecvInto(peer.WithName(w.Name), asMessage(w.RecvBuf))
		recvCount++
	}

	par := func(ranks []int, op func(plan.PeerID) error) error {
		errs := make([]error, len(ranks))
		var wg sync.WaitGroup
		for i, rank := range ranks {
			wg.Add(1)
			go func(i, rank int) {
				errs[i] = op(sess.peers[rank])
				wg.Done()
			}(i, rank)
		}
		wg.Wait()
		return mergeErrors(errs, "par")
	}

	seq := func(ranks []int, op func(plan.PeerID)) {
		for _, rank := range ranks {
			op(sess.peers[rank])
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
	return fmt.Errorf("%s failed with %s: %s", hint, utils.Pluralize(failed, "error", "errors"), msg)
}
