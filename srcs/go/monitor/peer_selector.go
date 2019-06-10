package monitor

import (
	"fmt"
	"container/heap"
	standard "github.com/lsds/KungFu/srcs/go/datastructures"
)

type MonitoringSelector struct {
	pqPtr *standard.PriorityQueue
	freq map[int]int64
}

func NewMonitoringSelector() *MonitoringSelector {
	ms :=  MonitoringSelector {
		pqPtr : standard.NewPriorityQueue(),
		freq : make(map[int]int64),
	}
	return &ms
}

func (ms *MonitoringSelector) RegisterRequest(toPeer int, requestLatency float64) {
	currFrequency := ms.freq[toPeer]
	ms.freq[toPeer] = currFrequency + int64(1)

	metric := &standard.AggregatedMetric {
		Rank : toPeer,
		Frequency : currFrequency + int64(1),
		Latency : requestLatency,
	}

	heap.Push(ms.pqPtr, metric)
}

func  (ms *MonitoringSelector) PickBestPeer(currentDestination int) int {
	 if ms.pqPtr.Len() == 0 {
		fmt.Printf("%s\n", "Priority queue  is empty, returning current peer")
		return currentDestination
	 } else {
		best := heap.Pop(ms.pqPtr).(*standard.AggregatedMetric)
		return best.Rank
	 }
}

