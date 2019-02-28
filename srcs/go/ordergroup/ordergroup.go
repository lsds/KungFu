package ordergroup

import (
	"log"
	"os"
	"sync"
)

type orderGroup struct {
	sync.Mutex

	size  int
	names []string
	ranks map[string]int
	dones []chan struct{}
}


func (names []string) *orderGroup {
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
		<-g.dones[rank]
		defer func() { g.dones[rank+1] <- struct{}{} }()
		f()
	}()
}

func (g *orderGroup) Start() {
	g.dones[0] <- struct{}{}
}

func (g *orderGroup) Wait() {
	<-g.dones[g.size]
}
