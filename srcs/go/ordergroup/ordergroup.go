package ordergroup

import (
	"log"
	"os"
)

type orderGroup struct {
	size  int
	names []string
	ranks map[string]int
	dones []chan struct{}
}

func New(names []string) *orderGroup {
	var dones []chan struct{}
	ranks := make(map[string]int)
	for i, name := range names {
		ranks[name] = i
		dones = append(dones, make(chan struct{}, 1))
	}
	dones = append(dones, make(chan struct{}, 1))
	g := &orderGroup{
		size:  len(names),
		names: names,
		ranks: ranks,
		dones: dones,
	}
	g.Start()
	return g
}

func (g *orderGroup) Do(name string, f func()) {
	rank, ok := g.ranks[name]
	if !ok {
		log.Printf("%s is not schedued", name)
		os.Exit(1)
	}
	go func() {
		g.wait(rank)
		defer g.start(rank + 1)
		f()
	}()
}

func (g *orderGroup) start(i int) {
	g.dones[i] <- struct{}{}
}

func (g *orderGroup) wait(i int) {
	<-g.dones[i]
}

func (g *orderGroup) Start() {
	g.start(0)
}

func (g *orderGroup) Wait() {
	g.wait(g.size)
}
