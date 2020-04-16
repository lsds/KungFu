package session

import (
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

//SmartAllReduce performs an optimized AllReduce operation over the given workspace parameter
//by monitoring the performance of different concurrently executed collective communications
//strategies and applying weights to optimize the choice between them based on the monitoring
func (sess *Session) SmartAllReduce(w kb.Workspace) error {
	return sess.runStrategies(w, plan.EvenPartition, sess.strategies)
}

func (sess *Session) runAdaptStrategiesWithWeightedHash(w kb.Workspace, p kb.PartitionFunc, strategies strategyList, strategyHash strategyHashFunc) error {
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
	return utils.MergeErrors(errs, "runStrategies")
}

func (sess *Session) runAdaptStrategies(w kb.Workspace, p kb.PartitionFunc, strategies strategyList) error {
	return sess.runAdaptStrategiesWithWeightedHash(w, p, strategies, sess.strategyHash)
}
