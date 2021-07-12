package session

import (
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/execution"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/graph"
	"github.com/lsds/KungFu/srcs/go/rchannel/client"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/rchannel/handler"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

const defaultRoot = 0

// Session contains the immutable peer list for a given period of logical duration
type Session struct {
	sync.Mutex

	localStrategies   strategyList
	globalStrategies  strategyList
	crossStrategies   strategyList
	self              plan.PeerID
	peers             plan.PeerList
	rank              int
	localRank         int
	localSize         int
	hostCount         int
	client            *client.Client
	collectiveHandler *handler.CollectiveEndpoint
	strategyHash      strategyHashFunc
	strategyStats     []StrategyStatSnapshot
}

func New(strategy kb.Strategy, self plan.PeerID, pl plan.PeerList, client *client.Client, collectiveHandler *handler.CollectiveEndpoint) (*Session, bool) {
	rank, ok := pl.Rank(self)
	if !ok {
		return nil, false
	}
	localRank, ok := pl.LocalRank(self)
	if !ok {
		return nil, false
	}
	if strategy == kb.Auto {
		strategy = autoSelect(pl)
	}
	sess := &Session{
		localStrategies:   genLocalStrategyList(pl),
		globalStrategies:  genGlobalStrategyList(pl, strategy),
		crossStrategies:   genCrossStrategyList(pl, strategy),
		self:              self,
		peers:             pl,
		rank:              rank,
		localRank:         localRank,
		localSize:         pl.LocalSize(self),
		hostCount:         pl.HostCount(),
		client:            client,
		collectiveHandler: collectiveHandler,
		strategyHash:      getStrategyHash(),
	}
	return sess, true
}

func (sess *Session) Size() int {
	return len(sess.peers)
}

func (sess *Session) Rank() int {
	return sess.rank
}

func (sess *Session) LocalRank() int {
	return sess.localRank
}

func (sess *Session) LocalSize() int {
	return sess.localSize
}

func (sess *Session) HostCount() int {
	return sess.hostCount
}

func (sess *Session) Peer(rank int) plan.PeerID {
	return sess.peers[rank]
}

func (sess *Session) Barrier() error {
	sess.Lock()
	defer sess.Unlock()
	return sess.barrier()
}

func (sess *Session) barrier() error {
	k := len(sess.peers)
	count := k * 1
	dtype := kb.U8
	w := kb.Workspace{
		SendBuf: kb.NewVector(count, dtype),
		RecvBuf: kb.NewVector(count, dtype),
		OP:      kb.SUM,
		Name:    "kungfu::barrier", // TODO: use tag
	}
	return sess.runStrategies(w, plan.EvenPartition, sess.globalStrategies)
}

func (sess *Session) Consensus(w kb.Workspace) error {
	ok, err := sess.BytesConsensus(w.SendBuf.Data, w.Name)
	if err != nil {
		return err
	}
	w.RecvBuf.AsI8()[0] = boolToInt8(ok)
	return nil
}

func (sess *Session) BytesConsensus(bs []byte, name string) (bool, error) {
	n := len(bs)
	{
		x := kb.NewVector(1, kb.I32)
		y := kb.NewVector(1, kb.I32)
		z := kb.NewVector(1, kb.I32)
		x.AsI32()[0] = int32(n)
		w1 := kb.Workspace{SendBuf: x, RecvBuf: y, OP: kb.MIN, Name: ":consensus:len:min:" + name}
		w2 := kb.Workspace{SendBuf: x, RecvBuf: z, OP: kb.MAX, Name: ":consensus:len:max:" + name}
		assert.OK(sess.AllReduce(w1))
		assert.OK(sess.AllReduce(w2))
		if !utils.BytesEq(y.Data, z.Data) {
			return false, nil
		}
	}
	if n == 0 {
		return true, nil
	}
	{
		x := &kb.Vector{Data: bs, Count: n, Type: kb.U8}
		y := kb.NewVector(n, kb.U8)
		z := kb.NewVector(n, kb.U8)
		w1 := kb.Workspace{SendBuf: x, RecvBuf: y, OP: kb.MIN, Name: ":consensus:min:" + name}
		w2 := kb.Workspace{SendBuf: x, RecvBuf: z, OP: kb.MAX, Name: ":consensus:max:" + name}
		assert.OK(sess.AllReduce(w1))
		assert.OK(sess.AllReduce(w2))
		if !utils.BytesEq(y.Data, z.Data) {
			return false, nil
		}
	}
	return true, nil
}

func (sess *Session) Reduce(w kb.Workspace) error {
	strategy := sess.globalStrategies[0] // Assuming len(sess.globalStrategies) > 0
	return sess.runGraphs(w, strategy.reduceGraph)
}

func (sess *Session) Broadcast(w kb.Workspace) error {
	strategy := sess.globalStrategies[0] // Assuming len(sess.globalStrategies) > 0
	return sess.runGraphs(w, strategy.bcastGraph)
}

func (sess *Session) Gather(w kb.Workspace) error {
	// TODO: validate input
	return sess.runGather(w)
}

func (sess *Session) LocalReduce(w kb.Workspace) error {
	strategy := sess.localStrategies[0] // len(sess.localStrategies) == 1
	return sess.runGraphs(w, strategy.reduceGraph)
}

func (sess *Session) LocalBroadcast(w kb.Workspace) error {
	strategy := sess.localStrategies[0] // len(sess.localStrategies) == 1
	return sess.runGraphs(w, strategy.bcastGraph)
}

func asMessage(b *kb.Vector) connection.Message {
	return connection.Message{
		Length: uint32(len(b.Data)),
		Data:   b.Data,
	}
}

func (sess *Session) runGather(w kb.Workspace) error {
	if sess.rank != defaultRoot {
		peer := sess.peers[defaultRoot]
		return sess.client.Send(peer.WithName(w.Name), w.SendBuf.Data, connection.ConnCollective, connection.NoFlag)
	}
	var wg sync.WaitGroup
	count := w.SendBuf.Count
	for rank, peer := range sess.peers {
		wg.Add(1)
		go func(rank int, peer plan.PeerID, recvBuf *kb.Vector) {
			if rank == sess.rank {
				recvBuf.CopyFrom(w.SendBuf)
			} else {
				m := sess.collectiveHandler.Recv(peer.WithName(w.Name))
				b := &kb.Vector{Data: m.Data, Count: recvBuf.Count, Type: recvBuf.Type}
				recvBuf.CopyFrom(b)
			}
			wg.Done()
		}(rank, peer, w.RecvBuf.Slice(count*rank, count*(rank+1)))
	}
	wg.Wait()
	return nil // FIXME: handle errors
}

func isIsolated(rank int, graphs ...*graph.Graph) bool {
	for _, g := range graphs {
		if !g.IsIsolated(rank) {
			return false
		}
	}
	return true
}

func (sess *Session) runGraphs(w kb.Workspace, graphs ...*graph.Graph) error {
	if w.IsEmpty() { // w.IsInplace will panic
		return nil
	}
	if isIsolated(sess.rank, graphs...) {
		w.Forward()
		return nil
	}

	var recvCount int
	effectiveBuffer := func() *kb.Vector {
		if recvCount > 0 || w.IsInplace() {
			return w.RecvBuf
		}
		return w.SendBuf
	}
	var sendOnto execution.PeerFunc = func(peer plan.PeerID) error {
		return sess.client.Send(peer.WithName(w.Name), effectiveBuffer().Data, connection.ConnCollective, connection.NoFlag)
	}
	var sendInto execution.PeerFunc = func(peer plan.PeerID) error {
		return sess.client.Send(peer.WithName(w.Name), effectiveBuffer().Data, connection.ConnCollective, connection.WaitRecvBuf)
	}

	var lock sync.Mutex
	var recvOnto execution.PeerFunc = func(peer plan.PeerID) error {
		m := sess.collectiveHandler.Recv(peer.WithName(w.Name))
		b := &kb.Vector{Data: m.Data, Count: w.SendBuf.Count, Type: w.SendBuf.Type}
		lock.Lock()
		defer lock.Unlock()
		kb.Transform2(w.RecvBuf, effectiveBuffer(), b, w.OP)
		recvCount++
		connection.PutBuf(m.Data) // Recycle buffer on the RecvOnto path
		return nil
	}

	var recvInto execution.PeerFunc = func(peer plan.PeerID) error {
		sess.collectiveHandler.RecvInto(peer.WithName(w.Name), asMessage(w.RecvBuf))
		recvCount++
		return nil
	}

	for _, g := range graphs {
		prevs := sess.peers.Select(g.Prevs(sess.rank))
		nexts := sess.peers.Select(g.Nexts(sess.rank))
		if g.IsSelfLoop(sess.rank) {
			if err := recvOnto.Par(prevs); err != nil {
				return err
			}
			if err := sendOnto.Par(nexts); err != nil {
				return err
			}
		} else {
			if len(prevs) > 1 {
				log.Errorf("more than once recvInto detected at node %d", sess.rank)
			}
			if len(prevs) == 0 && recvCount == 0 {
				w.Forward()
			} else {
				if err := recvInto.Seq(prevs); err != nil { // len(prevs) == 1 is expected
					return err
				}
			}
			if err := sendInto.Par(nexts); err != nil {
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

func (sess *Session) runStrategiesWithHash(w kb.Workspace, p kb.PartitionFunc, strategies strategyList, strategyHash strategyHashFunc) error {
	k := ceilDiv(w.RecvBuf.Count*w.RecvBuf.Type.Size(), chunkSize)
	errs := make([]error, k)
	var wg sync.WaitGroup
	for i, w := range w.Split(p, k) {
		wg.Add(1)
		go func(i int, w kb.Workspace, s strategy) {
			errs[i] = sess.runGraphs(w, s.reduceGraph, s.bcastGraph)
			wg.Done()
		}(i, w, strategies.choose(int(strategyHash(i, w.Name))))
	}
	wg.Wait()
	return utils.MergeErrors(errs, "runStrategiesWithHash")
}

func (sess *Session) runStrategies(w kb.Workspace, p kb.PartitionFunc, strategies strategyList) error {
	return sess.runStrategiesWithHash(w, p, strategies, sess.strategyHash)
}

func boolToInt8(v bool) int8 {
	if v {
		return 1
	}
	return 0
}
