package ordergroup

import (
	"sync"
	"sync/atomic"
)

type Option struct {
	AutoWait bool
}

type OrderGroup struct {
	size     int
	ready    []chan struct{}
	wg       sync.WaitGroup
	started  int32
	autoWait bool
}

func New(n int, opt Option) *OrderGroup {
	var ready []chan struct{}
	for i := 0; i <= n; i++ {
		ready = append(ready, make(chan struct{}, 1))
	}
	g := &OrderGroup{
		size:     n,
		ready:    ready,
		autoWait: opt.AutoWait,
	}
	g.wg.Add(g.size)
	g.Start()
	return g
}

func (g *OrderGroup) DoRank(rank int, f func()) {
	go func() {
		g.wait(rank)
		f()
		g.start(rank + 1)
		g.wg.Done()
	}()
	started := atomic.AddInt32(&g.started, 1)
	if int(started) == g.size && g.autoWait {
		g.Wait()
	}
}

func (g *OrderGroup) start(i int) {
	g.ready[i] <- struct{}{}
}

func (g *OrderGroup) wait(i int) {
	<-g.ready[i]
}

func (g *OrderGroup) Start() {
	g.start(0)
}

func (g *OrderGroup) Wait() {
	g.wg.Wait()
}
