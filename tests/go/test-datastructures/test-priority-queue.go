package main

import (
	"fmt"

	"container/heap"
	standard "github.com/lsds/KungFu/srcs/go/datastructures"

)


func main() {
	pqPtr := standard.NewPriorityQueue()

	m1 := &standard.AggregatedMetric {
		Rank : 1,
		Frequency : 20,
		Latency : 30,
		Index : 0,
	}

	m2 := &standard.AggregatedMetric {
		Rank : 2,
		Frequency : 30,
		Latency : 10,
		Index : 1,
	}

	m3 := &standard.AggregatedMetric {
		Rank : 3,
		Frequency : 30,
		Latency : 10,
		Index : 1,
	}


	m4 := &standard.AggregatedMetric {
		Rank : 4,
		Frequency : 30,
		Latency : 2,
		Index : 1,
	}

	heap.Push(pqPtr, m1)
	heap.Push(pqPtr, m2)
	

	fmt.Printf("First: %+v\n", heap.Pop(pqPtr).(*standard.AggregatedMetric))
	fmt.Printf("Second: %+v\n", heap.Pop(pqPtr).(*standard.AggregatedMetric))

	heap.Push(pqPtr, m4)
	heap.Push(pqPtr, m3)
	

	fmt.Printf("Third: %+v\n", heap.Pop(pqPtr).(*standard.AggregatedMetric))
	fmt.Printf("Fourth: %+v\n", heap.Pop(pqPtr).(*standard.AggregatedMetric))

}

