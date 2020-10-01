package session

import (
	"sync"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/client"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func (sess *Session) runMonitoredStrategiesWithHash(w kb.Workspace, p kb.PartitionFunc, strategies strategyList, strategyHash strategyHashFunc) error {
	k := ceilDiv(w.RecvBuf.Count*w.RecvBuf.Type.Size(), chunkSize)
	errs := make([]error, k)
	var wg sync.WaitGroup
	for i, w := range w.Split(p, k) {
		wg.Add(1)
		go func(i int, w kb.Workspace, s strategy) {
			startTime := time.Now()
			errs[i] = sess.runGraphs(w, s.reduceGraph, s.bcastGraph)
			endTime := time.Now()
			s.stat.Update(startTime, endTime, w.SendBuf.Count*w.SendBuf.Type.Size())
			wg.Done()
		}(i, w, strategies.choose(int(strategyHash(i, w.Name))))
	}
	wg.Wait()
	return utils.MergeErrors(errs, "runMonitoredStrategiesWithHash")
}

func (sess *Session) runMonitoredStrategies(w kb.Workspace, p kb.PartitionFunc, strategies strategyList) error {
	return sess.runMonitoredStrategiesWithHash(w, p, strategies, sess.strategyHash)
}

// GetPeerLatencies is deprecated
func (sess *Session) GetPeerLatencies() []time.Duration {
	results := make([]time.Duration, len(sess.peers))
	var wg sync.WaitGroup
	for rank, peer := range sess.peers {
		if rank != sess.rank {
			wg.Add(1)
			go func(rank int, peer plan.PeerID) {
				results[rank] = getLatency(sess.self, peer)
				wg.Done()
			}(rank, peer)
		} else {
			results[rank] = 0
		}
	}
	wg.Wait()
	return results
}

func getLatency(self, peer plan.PeerID) time.Duration {
	client := client.New(self, config.UseUnixSock)
	d, err := client.Ping(peer)
	if err != nil {
		log.Errorf("ping(%s) failed, error ignored!", peer)
		// TODO handle error
	}
	return d
}

func (sess *Session) GetEgressRates() []float64 {
	addrs := make([]plan.NetAddr, len(sess.peers))
	for i, p := range sess.peers {
		addrs[i] = plan.NetAddr(p)
	}
	return sess.client.GetEgressRates(addrs)
}
