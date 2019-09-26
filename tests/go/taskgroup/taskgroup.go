package taskgroup

import "sync"

type Group struct {
	tasks []func()
}

func (g *Group) Add(f func()) {
	g.tasks = append(g.tasks, f)
}

func (g *Group) Par() {
	var wg sync.WaitGroup
	for _, t := range g.tasks {
		wg.Add(1)
		go func(t func()) {
			t()
			wg.Done()
		}(t)
	}
	wg.Wait()
}

func (g *Group) Seq() {
	for _, t := range g.tasks {
		t()
	}
}
