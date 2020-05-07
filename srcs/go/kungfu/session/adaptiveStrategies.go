package session

import (
	"fmt"
	"sync"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/tests/go/testutils"
)

const (
	interferenceThreshold = 1.5
)

//SmartAllReduce performs an optimized AllReduce operation over the given workspace parameter
//by monitoring the performance of different concurrently executed collective communications
//strategies and applying weights to optimize the choice between them based on the monitoring
func (sess *Session) SmartAllReduce(w kb.Workspace) error {
	return sess.runAdaptStrategies(w, plan.EvenPartition, sess.strategies)
}

func (sess *Session) runAdaptStrategiesWithWeightedHash(w kb.Workspace, p kb.PartitionFunc, strategies strategyList, strategyHash strategyHashFunc) error {
	k := ceilDiv(w.RecvBuf.Count*w.RecvBuf.Type.Size(), chunkSize)
	errs := make([]error, k)
	var wg sync.WaitGroup
	for i, w := range w.Split(p, k) {
		//fmt.Println("DEV::RunningAdaptStrategies::Strategy=", strategies.choose(int(strategyHash(i, w.Name))))
		wg.Add(1)
		go func(i int, w kb.Workspace, s strategy) {
			var dur time.Duration
			stpWatch := testutils.NewStopWatch()
			errs[i] = sess.runGraphs(w, s.reduceGraph, s.bcastGraph)
			stpWatch.StopAndSave(&dur)
			s.stat.Update(dur)
			//fmt.Println("DEV::Iter::", i, "::Duration::", dur, "::SessStrategyDur::", s.duration)
			wg.Done()
		}(i, w, strategies.choose(int(strategyHash(i, w.Name))))
	}
	wg.Wait()
	return utils.MergeErrors(errs, "runStrategies")
}

func (sess *Session) runAdaptStrategies(w kb.Workspace, p kb.PartitionFunc, strategies strategyList) error {
	return sess.runAdaptStrategiesWithWeightedHash(w, p, strategies, sess.strategyHash)
}

//LogStats reports Stat object for a specific strategy
func (sess *Session) LogStats(stratIdx int) StrategyStat {
	return *sess.strategies[stratIdx].stat
}

func (sess *Session) PrintSessionState() {
	fmt.Println("Printing current state of session strategies")
	fmt.Println("Available strategies: ", len(sess.strategies))

	for i, s := range sess.strategies {
		fmt.Println("Strategy #", i, " Master [", s.bcastGraph.Master, "] avgDuration=", s.stat.AvgDuration, " CMA=", s.stat.CmaDuration)
	}
}

//MonitorStrategy examines whether the current communication strategy is experiencing
//communication interference. It should not be invoked while the strategy is still
//used in background communication operations. No locking of the underlying data
//structure is used.
func (sess *Session) MonitorStrategy(buff []int8) {
	s := sess.strategies[0]
	buff[0] = 0
	if s.stat.AvgDuration > time.Duration((interferenceThreshold * float64(s.stat.refWindow.AvgDuration))) {
		buff[0] = 1
	}
}

func (sess *Session) MonitorMultipleStrategies(buff []int8) bool {

	var count int
	for _, s := range sess.strategies {
		if !s.stat.suspended {
			count++
		}
	}

	fmt.Println("MonitorStrategies: number of active strategies:", count)

	//if only 1 active strategy, don't do anything
	if count < 2 {
		return false
	}

	for i := range buff {
		buff[i] = 0
	}

	var change bool = false

	//TODO: find more efficient way of doing this
	for i, s := range sess.strategies {
		var resAvg time.Duration
		var resCount int
		for j, ss := range sess.strategies {
			if i == j || ss.stat.suspended {
				continue
			}
			resAvg += ss.stat.AvgDuration
			resCount++
		}
		resAvg = time.Duration(float64(resAvg) / float64(resCount))

		if s.stat.AvgDuration > time.Duration((interferenceThreshold * float64(resAvg))) {
			//flag the strategy as deactivated
			//s.stat.suspended = true
			change = true
			buff[i] = 1
			fmt.Println("ATTENTION: Strategy #", i, " has been proposed for suspension due to detected communication overhead")
		}
	}

	return change
}

func (sess *Session) ChangeStrategies(buff []int8) {
	//TODO: volatile. check that not all strategies will
	//be suspended all together.

	for i := range buff {
		if buff[i] > int8(sess.Size()) {
			//reached consensus
			fmt.Println("Session:: reached consensus on suspending strategy #", i)

			sess.strategies[i].stat.suspended = true
		}
	}
}

func (sess *Session) ChangeStrategy(buff []int8) {
	//TODO: volatile. check that not all strategies will
	//be suspended all together.

	if buff[0] > int8(sess.Size()) {
		//reached consensus
		fmt.Println("Session:: reached consensus on changing strategy #", 0)

		fmt.Println("Session:: switching to alternative strategy")

		panic("Not implemented yet")
	}
}

//GetNumStrategies returns the number of different strategies
//for a given session
func (sess *Session) GetNumStrategies() int {
	return len(sess.strategies)
}
