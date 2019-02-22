package monitor

import (
	"fmt"
	"io"
	"sync"
	"sync/atomic"
)

type accumulator struct {
	name  string
	value int64
}

func newAccumulator(name string) *accumulator {
	return &accumulator{
		name: name,
	}
}

func (a *accumulator) Add(n int64) int64 {
	return atomic.AddInt64(&a.value, n)
}

func (a *accumulator) Get() int64 {
	return atomic.LoadInt64(&a.value)
}

func (a *accumulator) WriteTo(w io.Writer) {
	val := atomic.LoadInt64(&a.value)
	// FIXME: add labels
	fmt.Fprintf(w, "%s{} %d\n", a.name, val)
}

type rate struct {
	sync.Mutex

	name   string
	prev   int64
	target *accumulator
	value  int64
}

func newRate(a *accumulator, suffixs string) *rate {
	r := &rate{
		name:   a.name + suffixs,
		target: a,
	}
	return r
}

func (r *rate) update() {
	now := r.target.Get()
	r.Lock()
	defer r.Unlock()
	r.value = now - r.prev
	r.prev = now
}

func (r *rate) WriteTo(w io.Writer) {
	r.Lock()
	defer r.Unlock()
	// FIXME: add labels
	fmt.Fprintf(w, "%s{} %d\n", r.name, r.value)
}
