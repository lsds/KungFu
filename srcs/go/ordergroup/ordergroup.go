package ordergroup

import (
	"log"
	"os"
	"sync"
)

type OrderGroup struct {
	sync.Mutex

	size  int
	names []string
	ranks map[string]int
	dones []chan struct{}
}

func NewGroup(names []string) *OrderGroup {
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
	go func() {
		<-g.dones[rank]
		defer func() { g.dones[rank+1] <- struct{}{} }()
		f()
	}()
}

func (g *OrderGroup) Start() {
	g.dones[0] <- struct{}{}
}

func (g *OrderGroup) Wait() {
	<-g.dones[g.size]
}
