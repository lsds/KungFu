package ordergroup

import (
	"log"
	"os"
)

type OrderGroup struct {
	size  int
	names []string
	ranks map[string]int
	dones []chan struct{}
}

func NewRanked(n int) *OrderGroup {
	var dones []chan struct{}
	for i := 0; i <= n; i++ {
		dones = append(dones, make(chan struct{}, 1))
	}
	g := &OrderGroup{
		size:  n,
		dones: dones,
	}
	g.Start()
	return g
}

func New(names []string) *OrderGroup {
	var dones []chan struct{}
	ranks := make(map[string]int)
	for i, name := range names {
		ranks[name] = i
		dones = append(dones, make(chan struct{}, 1))
	}
	dones = append(dones, make(chan struct{}, 1))
	g := &OrderGroup{
		size:  len(names),
		names: names,
		ranks: ranks,
		dones: dones,
	}
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
		defer g.start(rank + 1)
		f()
	}()
}

func (g *OrderGroup) start(i int) {
	g.dones[i] <- struct{}{}
}

func (g *OrderGroup) wait(i int) {
	<-g.dones[i]
}

func (g *OrderGroup) Start() {
	g.start(0)
}

func (g *OrderGroup) Wait() {
	g.wait(g.size)
}
