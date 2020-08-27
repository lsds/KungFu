package session

import (
	"fmt"
	"sync"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/tests/go/fakemodel"
)

const (
	interferenceThreshold = 0.8
	alternativeStrategy   = 1
)

//SmartAllReduce performs an optimized AllReduce operation over the given workspace parameter
//by monitoring the performance of different concurrently executed collective communications
//strategies and applying weights to optimize the choice between them based on the monitoring
func (sess *Session) SmartAllReduce(w kb.Workspace) error {
	return sess.runMonitoredStrategies(w, plan.EvenPartition, sess.globalStrategies)
}

func (sess *Session) runMonitoredStrategiesWithHash(w kb.Workspace, p kb.PartitionFunc, strategies strategyList, strategyHash strategyHashFunc) error {
	k := ceilDiv(w.RecvBuf.Count*w.RecvBuf.Type.Size(), chunkSize)
	errs := make([]error, k)
	var wg sync.WaitGroup
	for i, w := range w.Split(p, k) {
		wg.Add(1)
		go func(i int, w kb.Workspace, s strategy) {
			var startTime, endTime time.Time
			startTime = time.Now()
			errs[i] = sess.runGraphs(w, s.reduceGraph, s.bcastGraph)
			endTime = time.Now()
			s.stat.Update(startTime, endTime, w.SendBuf.Count*w.SendBuf.Type.Size())
			wg.Done()
		}(i, w, strategies.choose(int(strategyHash(i, w.Name))))
	}
	wg.Wait()
	return utils.MergeErrors(errs, "runStrategies")
}

func (sess *Session) runMonitoredStrategies(w kb.Workspace, p kb.PartitionFunc, strategies strategyList) error {
	return sess.runMonitoredStrategiesWithHash(w, p, strategies, sess.strategyHash)
}

//LogStats reports a Stat object for a specific strategy
//by appending it in the
func (sess *Session) LogStats(stratIdx int) {

	//TODO: fix this. Temporary for experiments only
	if stratIdx == -1 {
		sess.globalStrategies[0].stat.Reset()
		return
	}

	//calculate Throughput
	stats := sess.globalStrategies[stratIdx].stat
	t := float64(stats.accSize) / stats.lastEnd.Sub(*stats.firstBegin).Seconds() //time.Duration(stats.lastEnd-*stats.firstBegin)
	stats.Throughput = t

	// if sess.rank == 0 {
	// 	fmt.Println("LogStats: AccData=", testutils.ShowSize(stats.accSize), " Dur=", stats.lastEnd.Sub(*stats.firstBegin).Seconds(), " sec")
	// 	fmt.Println("LogStats: Throughput=", utils.ShowRate(stats.Throughput))
	// }

	//reset counters
	stats.Reset()

	sess.strategyStats = append(sess.strategyStats, sess.globalStrategies[stratIdx].stat.GetSnapshot())
}

//PrintStategyStats prints the Strategy Stats Snapshots that have been logged in the `sess.strategyStats` slice
func (sess *Session) PrintStategyStats() {
	fmt.Println("Printing current state of session strategies")
	fmt.Println("Available strategies: ", len(sess.globalStrategies))

	// for i, s := range sess.strategies {
	// 	fmt.Println("Strategy #", i, ",Master[", s.bcastGraph.Master, "],avgDuration=", s.stat.AvgDuration, ",CMA=", s.stat.CmaDuration)
	// }

	for i, ss := range sess.strategyStats {
		fmt.Println("Global Step #", i, ",Master[", 0, "],Throughput=", utils.ShowRate(ss.Throughput))
	}
}

//ChangeStrategy monitores throughput performance of the active strategy (strategy #0)
//and if it detects significan perfromance degredation, suspends the active strategy and
//changes to an alternate strategy.
func (sess *Session) ChangeStrategy() bool {
	var ret bool

	s := sess.globalStrategies[0]

	//s.stat.calcAvgWind()

	if s.stat.reff.Throughput == 0 {

		//TODO: temp dev change. make this permanent by considering distrib state
		// s.stat.reff.AvgDuration = s.stat.AvgDuration
		// s.stat.reff.CmaDuration = s.stat.CmaDuration
		// s.stat.reff.AvgWndDuration = s.stat.AvgWndDuration
		s.stat.reff.Throughput = s.stat.Throughput

		if sess.rank == 0 {
			fmt.Println("DEBUG:: Taking reff window snapshot")
			// fmt.Println("DEBUG:: AvgDur = ", s.stat.reff.AvgDuration)
			// fmt.Println("DEBUG:: AvgWndDur = ", s.stat.reff.AvgWndDuration)
			// fmt.Println("DEBUG:: CmaDur = ", s.stat.reff.CmaDuration)
			fmt.Println("DEBUG:: Thrroughput = ", utils.ShowRate(s.stat.reff.Throughput))
		}
		return false
	}

	db := fakemodel.NewDoubleBuffer(kb.I8, sess.GetNumStrategies())

	sb := db.SendBuf.AsI8()
	rb := db.RecvBuf.AsI8()

	sb[0] = 0

	if sess.rank == 0 {
		fmt.Println("MonitorStrategy:: Checking Throughput = ", utils.ShowRate(s.stat.Throughput), " reff = ", utils.ShowRate((interferenceThreshold * s.stat.reff.Throughput)))
	}

	if s.stat.Throughput < (interferenceThreshold * float64(s.stat.reff.Throughput)) {
		// fmt.Println("MonitorStrategy:: Congestion detected.")
		sb[0] = 1
	}

	w := kb.Workspace{
		SendBuf: db.SendBuf,
		RecvBuf: db.RecvBuf,
		OP:      kb.SUM,
		Name:    "StratMon",
	}

	// fmt.Println("DEBUG:: about to synch strategies mon, sending ", db.SendBuf.AsI8()[0])
	err := sess.AllReduce(w)
	if err != nil {
		utils.ExitErr(fmt.Errorf("%s failed performing allreduce", `Session.ChangeStrategy()`))
	}

	if sess.rank == 0 {
		fmt.Println("DEBUG:: monitoring synced")
		fmt.Println("ChangeStrategy:: rcved from cluster ", rb[0])
	}

	if rb[0] > int8(sess.Size()/2) {
		//reached consensus

		if sess.rank == 0 {
			fmt.Println("Session:: reached consensus on changing strategy #", 0)

			fmt.Println("Session:: switching to alternative strategy")
		}

		bcastGraph := plan.GenAlternativeStar(sess.peers, alternativeStrategy)
		ss := simpleStrategy(bcastGraph)

		ss.stat.reff = sess.globalStrategies[0].stat.reff
		sess.globalStrategies[0] = ss

		// fmt.Println("Session:: Switched to alternative strategy with master offset ", alternativeStrategy)

		ret = true
	}

	return ret
}

//GetNumStrategies returns the number of different strategies
//for a given session
func (sess *Session) GetNumStrategies() int {
	return len(sess.globalStrategies)
}
