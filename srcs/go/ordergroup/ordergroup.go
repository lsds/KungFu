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
	size     int
	tasks    []Task
	ready    chan int
	isReady  []int32
	wg       sync.WaitGroup
	started  int32
	autoWait bool
}

// New creates an OrderGroup of given size.
func New(n int, opt Option) *OrderGroup {
	g := &OrderGroup{
		size:     n,
		tasks:    make([]Task, n),
		ready:    make(chan int, n),
		isReady:  make([]int32, n),
		autoWait: opt.AutoWait,
	}
	g.wg.Add(g.size)
	g.Start()
	return g
}

// DoRank starts the i-th (0 <= i < n) rank.
func (g *OrderGroup) DoRank(i int, f Task) {
	g.tasks[i] = f
	atomic.StoreInt32(&g.isReady[i], 1)
	g.ready <- i
	started := atomic.AddInt32(&g.started, 1)
	if int(started) == g.size && g.autoWait {
		g.Wait()
	}
}

func (g *OrderGroup) Start() {
	go g.schedule()
}

func (g *OrderGroup) schedule() {
	var next int
	for range g.ready {
		for next < g.size {
			if isReady := atomic.LoadInt32(&g.isReady[next]); isReady == 0 {
				break
			}
			g.tasks[next]()
			next++
			g.wg.Done()
		}
	}
}

func (g *OrderGroup) Wait() {
	g.wg.Wait()
}

func (g *OrderGroup) Stop() {
	close(g.ready)
}
