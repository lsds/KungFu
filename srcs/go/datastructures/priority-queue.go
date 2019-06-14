// Go Priority Queue Implementation
// https://golang.org/pkg/container/heap/
package datastructures

import (
	"container/heap"
)

type AggregatedMetric struct {
	Rank      int
	Frequency int64
	Latency   float64
	Index     int
}

type PriorityQueue []*AggregatedMetric

// High priority means low latency and small frequency
func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].Frequency < pq[j].Frequency || (pq[i].Frequency == pq[j].Frequency && pq[i].Latency < pq[j].Latency)
}

func NewPriorityQueue() *PriorityQueue {
	pq := make(PriorityQueue, 0)
	heap.Init(&pq)
	return &pq
}

func (pq PriorityQueue) Len() int {
	return len(pq)
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].Index = i
	pq[j].Index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*AggregatedMetric)
	item.Index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.Index = -1
	*pq = old[0 : n-1]
	return item
}
