package monitor

import (
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"
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
	fmt.Fprintf(w, "%s %d\n", a.name, val)
}

type rate struct {
	sync.Mutex

	name   string
	prev   int64
	target *accumulator
	value  float64
}

func newRate(a *accumulator, name string) *rate {
	r := &rate{
		name:   name,
		target: a,
	}
	return r
}

const (
	totalUnitSuffix = `bytes`
	rateUnitSuffix  = `bytes_per_sec`
	rateTimeUnit    = float64(time.Second)
)

func (r *rate) update(p time.Duration) {
	now := r.target.Get()
	r.Lock()
	defer r.Unlock()
	r.value = float64(now-r.prev) / (float64(p) / rateTimeUnit)
	r.prev = now
}

func (r *rate) WriteTo(w io.Writer) {
	r.Lock()
	defer r.Unlock()
	fmt.Fprintf(w, "%s %f\n", r.name, r.value)
}

type rateAccumulator struct {
	a *accumulator
	r *rate
}

func newRateAccumulator(prefix string) *rateAccumulator {
	a := newAccumulator(prefix + "_total_" + totalUnitSuffix)
	r := newRate(a, prefix+"_rate_"+rateUnitSuffix)
	return &rateAccumulator{
		a: a,
		r: r,
	}
}

func (c *rateAccumulator) WriteTo(w io.Writer) {
	c.a.WriteTo(w)
	c.r.WriteTo(w)
}
