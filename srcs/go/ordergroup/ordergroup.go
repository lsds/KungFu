package ordergroup

import (
	"sync"
	"sync/atomic"
)

type Task func()

type Option struct {
	AutoWait bool
}

// OrderGroup ensures a set of async tasks are performed in scheduled order.
type OrderGroup struct {
	size        int
	tasks       []Task
	ready       chan int
	isReady     []bool
	allDone     sync.WaitGroup
	started     int32
	autoWait    bool
	arriveOrder []int
}

// New creates an OrderGroup of given size.
func New(n int, opt Option) *OrderGroup {
	g := &OrderGroup{
		size:     n,
		tasks:    make([]Task, n),
		ready:    make(chan int, n),
		isReady:  make([]bool, n),
		autoWait: opt.AutoWait,
	}
	g.allDone.Add(1)
	g.Start()
	return g
}

// DoRank starts the i-th (0 <= i < n) rank.
func (g *OrderGroup) DoRank(i int, f Task) {
	g.tasks[i] = f
	g.ready <- i
	if started := atomic.AddInt32(&g.started, 1); int(started) == g.size && g.autoWait {
		g.Wait()
	}
}

func (g *OrderGroup) Start() {
	go g.schedule()
}

func (g *OrderGroup) schedule() {
	var arriveOrder []int
	var next int
	for i := range g.ready {
		arriveOrder = append(arriveOrder, i)
		g.isReady[i] = true
		for next < g.size {
			if !g.isReady[next] {
				break
			}
			g.tasks[next]()
			next++
		}
		if next == g.size {
			break
		}
	}
	g.arriveOrder = arriveOrder
	g.allDone.Done()
}

func (g *OrderGroup) Wait() []int {
	g.allDone.Wait()
	return g.arriveOrder
}

func (g *OrderGroup) Stop() {
	close(g.ready)
}
