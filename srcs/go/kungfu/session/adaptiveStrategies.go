package session

import (
	"fmt"
	"sync"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/log"
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

//CalcStats reports a Stat object for the current active strategyt
func (sess *Session) CalcStats() {

	if len(sess.globalStrategies) != 1 {
		log.Errorf("CalcStats should only be called with one active communication strategy")
		return
	}

	//calculate Throughput
	stats := sess.globalStrategies[0].stat

	//avoid first invocation before first training step, thus no data monitored
	if stats.accSize == 0 {
		return
	}

	t := float64(stats.accSize) / stats.lastEnd.Sub(*stats.firstBegin).Seconds() //time.Duration(stats.lastEnd-*stats.firstBegin)
	stats.Throughput = t

	// if sess.rank == 0 {
	// 	fmt.Println("CalcStats: AccData=", testutils.ShowSize(int64(stats.accSize)), " Dur=", stats.lastEnd.Sub(*stats.firstBegin).Seconds(), " sec")
	// 	fmt.Println("CalcStats: Throughput=", utils.ShowRate(stats.Throughput))
	// }

	//reset counters
	stats.Reset()
}

//LogStats stores a snapshot of the `StrategyStat` object for the current active communication strategy
func (sess *Session) LogStats() {

	if len(sess.globalStrategies) != 1 {
		log.Errorf("LogStats should only be called with one active communication strategy")
		return
	}

	sess.strategyStats = append(sess.strategyStats, sess.globalStrategies[0].stat.GetSnapshot())
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

//CheckInterference is checking current state of the strategy stat metrics against the
//reference window and communicates with the cluster whether to change to an alternate
//communication strategy or not. Returns true if the cluster reached consensus on changing,
//flase otherwise.
func (sess *Session) CheckInterference() bool {
	var ret = false

	if len(sess.globalStrategies) != 1 {
		log.Errorf("CheckInterference should only be called with one active communication strategy")
		return ret
	}

	s := sess.globalStrategies[0]

	if s.stat.reff.Throughput == 0 {

		s.stat.reff.Throughput = s.stat.Throughput

		if sess.rank == 0 {
			log.Infof("AP:: Taking reff window snapshot")
			log.Infof("AP:: Thrroughput = %s", utils.ShowRate(s.stat.reff.Throughput))
		}
		return ret
	}

	//communicating metrics with cluster for reaching consensus
	db := fakemodel.NewDoubleBuffer(kb.I8, sess.GetNumStrategies())

	sb := db.SendBuf.AsI8()
	rb := db.RecvBuf.AsI8()

	sb[0] = 0

	if sess.rank == 0 {
		log.Infof("MonitorStrategy:: Checking Throughput = %s, reff= %s",
			utils.ShowRate(s.stat.Throughput),
			utils.ShowRate((interferenceThreshold * s.stat.reff.Throughput)))
	}

	//check for congestion
	if s.stat.Throughput < (interferenceThreshold * float64(s.stat.reff.Throughput)) {
		sb[0] = 1
	}

	w := kb.Workspace{
		SendBuf: db.SendBuf,
		RecvBuf: db.RecvBuf,
		OP:      kb.SUM,
		Name:    "StratMon",
	}

	err := sess.AllReduce(w)
	if err != nil {
		utils.ExitErr(fmt.Errorf("%s failed performing allreduce", `Session.CheckInterference()`))
	}

	if sess.rank == 0 {
		log.Infof("AP:: cluster response -> %d", rb[0])
	}

	if rb[0] > int8(sess.Size()/2) {
		//reached consensus
		if sess.rank == 0 {
			log.Infof("AP:: cluster reached consensus on changing to alternative strategy")
		}
		ret = true
	}

	return ret
}

//GetNumStrategies returns the number of different strategies
//for a given session
func (sess *Session) GetNumStrategies() int {
	return len(sess.globalStrategies)
}
