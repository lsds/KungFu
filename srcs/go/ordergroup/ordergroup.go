package ordergroup

import (
	"log"
	"os"
	"sync"
	"sync/atomic"
)

type Option struct {
	AutoWait bool
}

type OrderGroup struct {
	size     int
	names    []string
	ranks    map[string]int
	ready    []chan struct{}
	wg       sync.WaitGroup
	started  int32
	autoWait bool
}

func NewRanked(n int, opt Option) *OrderGroup {
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

func New(names []string) *OrderGroup {
	var ready []chan struct{}
	ranks := make(map[string]int)
	for i, name := range names {
		ranks[name] = i
		ready = append(ready, make(chan struct{}, 1))
	}
	ready = append(ready, make(chan struct{}, 1))
	g := &OrderGroup{
		size:  len(names),
		names: names,
		ranks: ranks,
		ready: ready,
	}
	g.wg.Add(g.size)
	g.Start()
	return g
}

func (g *OrderGroup) Do(name string, f func()) {
	rank, ok := g.ranks[name]
	if !ok {
		log.Printf("%s is not schedued", name)
		os.Exit(1)
	}
	g.DoRank(rank, f)
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
