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
	interferenceThreshold = 1.2
	alternativeStrategy   = 1
)

//SmartAllReduce performs an optimized AllReduce operation over the given workspace parameter
//by monitoring the performance of different concurrently executed collective communications
//strategies and applying weights to optimize the choice between them based on the monitoring
func (sess *Session) SmartAllReduce(w kb.Workspace) error {
	return sess.runMonitoresStrategies(w, plan.EvenPartition, sess.strategies)
}

func (sess *Session) runMonitoredStrategiesWithHash(w kb.Workspace, p kb.PartitionFunc, strategies strategyList, strategyHash strategyHashFunc) error {
	k := ceilDiv(w.RecvBuf.Count*w.RecvBuf.Type.Size(), chunkSize)
	errs := make([]error, k)
	var wg sync.WaitGroup
	for i, w := range w.Split(p, k) {
		wg.Add(1)
		go func(i int, w kb.Workspace, s strategy) {
			var startTime, endTime time.Time
			// var dur time.Duration
			// stpWatch := testutils.NewStopWatch()
			startTime = time.Now()
			errs[i] = sess.runGraphs(w, s.reduceGraph, s.bcastGraph)
			endTime = time.Now()
			// stpWatch.StopAndSave(&dur)
			s.stat.Update(startTime, endTime, w.SendBuf.Count*2)
			wg.Done()
		}(i, w, strategies.choose(int(strategyHash(i, w.Name))))
	}
	wg.Wait()
	return utils.MergeErrors(errs, "runStrategies")
}

func (sess *Session) runMonitoresStrategies(w kb.Workspace, p kb.PartitionFunc, strategies strategyList) error {
	return sess.runMonitoredStrategiesWithHash(w, p, strategies, sess.strategyHash)
}

//LogStats reports Stat object for a specific strategy
func (sess *Session) LogStats(stratIdx int) {

	//calculate Throughput
	stats := sess.strategies[stratIdx].stat
	t := float64(stats.accSize) / stats.lastEnd.Sub(*stats.firstBegin).Seconds() //time.Duration(stats.lastEnd-*stats.firstBegin)
	stats.Throughput = t

	//reset counters
	stats.Reset()

	sess.strategyStats = append(sess.strategyStats, sess.strategies[stratIdx].stat.GetSnapshot())

	if sess.rank == 0 {
		fmt.Println("LogStats: Throughput=", utils.ShowRate(stats.Throughput))
	}
}

func (sess *Session) PrintStategyStats() {
	fmt.Println("Printing current state of session strategies")
	fmt.Println("Available strategies: ", len(sess.strategies))

	// for i, s := range sess.strategies {
	// 	fmt.Println("Strategy #", i, ",Master[", s.bcastGraph.Master, "],avgDuration=", s.stat.AvgDuration, ",CMA=", s.stat.CmaDuration)
	// }

	for i, ss := range sess.strategyStats {
		fmt.Println("epoch #", i, ",Master[", 0, "],Throughput=", utils.ShowRate(ss.Throughput))
	}
}

//MonitorStrategy examines whether the current communication strategy is experiencing
//communication interference. It should not be invoked while the strategy is still
//used in background communication operations. No locking of the underlying data
//structure is used.
// func (sess *Session) MonitorStrategy(buff []int8) {
// 	s := sess.strategies[0]
// 	buff[0] = 0
// 	fmt.Println("MonitorStrategy:: Checking AvgDur = ", s.stat.AvgDuration, " reff = ", time.Duration((interferenceThreshold * float64(s.stat.reff.AvgDuration))))
// 	if s.stat.AvgDuration > time.Duration((interferenceThreshold * float64(s.stat.reff.AvgDuration))) {
// 		fmt.Println("MonitorStrategy:: Congestion detected.")
// 		buff[0] = 1
// 	}
// }

// func (sess *Session) MonitorMultipleStrategies(buff []int8) bool {

// 	var count int
// 	for _, s := range sess.strategies {
// 		if !s.stat.suspended {
// 			count++
// 		}
// 	}

// 	fmt.Println("MonitorStrategies: number of active strategies:", count)

// 	//if only 1 active strategy, don't do anything
// 	if count < 2 {
// 		return false
// 	}

// 	for i := range buff {
// 		buff[i] = 0
// 	}

// 	var change bool = false

// 	//TODO: find more efficient way of doing this
// 	for i, s := range sess.strategies {
// 		var resAvg time.Duration
// 		var resCount int
// 		for j, ss := range sess.strategies {
// 			if i == j || ss.stat.suspended {
// 				continue
// 			}
// 			resAvg += ss.stat.AvgDuration
// 			resCount++
// 		}
// 		resAvg = time.Duration(float64(resAvg) / float64(resCount))

// 		if s.stat.AvgDuration > time.Duration((interferenceThreshold * float64(resAvg))) {
// 			//flag the strategy as deactivated
// 			//s.stat.suspended = true
// 			change = true
// 			buff[i] = 1
// 			fmt.Println("ATTENTION: Strategy #", i, " has been proposed for suspension due to detected communication overhead")
// 		}
// 	}

// 	return change
// }

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

// func (sess *Session) ChangeStrategy(buff []int8, off int) bool {
// 	//TODO: volatile. check that not all strategies will
// 	//be suspended all together.
// 	var ret bool

// 	fmt.Println("ChangeStrategy:: rcved from cluster ", buff[0])

// 	if buff[0] > int8(sess.Size()/2) {
// 		//reached consensus
// 		fmt.Println("Session:: reached consensus on changing strategy #", 0)

// 		fmt.Println("Session:: switching to alternative strategy")

// 		bcastGraph := plan.GenAlternativeStar(sess.peers, off)
// 		reduceGraph := plan.GenDefaultReduceGraph(bcastGraph)
// 		ss := strategy{
// 			reduceGraph: reduceGraph,
// 			bcastGraph:  bcastGraph,
// 			stat:        &StrategyStat{},
// 		}

// 		ss.stat.refWindow = sess.strategies[0].stat.refWindow
// 		sess.strategies[0] = ss

// 		fmt.Println("Session:: Switched to alternative strategy with master offset ", off)

// 		ret = true
// 	}

// 	return ret
// }

func (sess *Session) ChangeStrategy() bool {
	//TODO: volatile. check that not all strategies will
	//be suspended all together.
	var ret bool

	s := sess.strategies[0]

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
			fmt.Println("DEBUG:: Thrroughput = ", s.stat.reff.Throughput)
		}
		return false
	}

	db := fakemodel.NewDoubleBuffer(kb.I8, sess.GetNumStrategies())

	sb := db.SendBuf.AsI8()
	rb := db.RecvBuf.AsI8()

	sb[0] = 0

	if sess.rank == 0 {
		fmt.Println("MonitorStrategy:: Checking Throughput = ", s.stat.Throughput, " reff = ", utils.ShowRate((interferenceThreshold * s.stat.reff.Throughput)))
	}

	if s.stat.Throughput > (interferenceThreshold * float64(s.stat.reff.Throughput)) {
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
		reduceGraph := plan.GenDefaultReduceGraph(bcastGraph)
		ss := strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
			stat:        &StrategyStat{},
		}

		ss.stat.reff = sess.strategies[0].stat.reff
		sess.strategies[0] = ss

		// fmt.Println("Session:: Switched to alternative strategy with master offset ", alternativeStrategy)

		ret = true
	}

	return ret
}

//GetNumStrategies returns the number of different strategies
//for a given session
func (sess *Session) GetNumStrategies() int {
	return len(sess.strategies)
}

// func (stat *StrategyStat) calcAvgWind() {
// 	var acc, ret time.Duration

// 	for _, val := range stat.AvgWnd {
// 		acc = acc + val
// 	}

// 	d := stat.wndBack - stat.wndFront
// 	if d > 0 {
// 		ret = time.Duration(float64(acc) / float64(d))
// 	} else {
// 		ret = time.Duration(float64(acc) / float64(avgWndCapacity))
// 	}

// 	stat.AvgWndDuration = ret
// }
