package monitor

import (
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
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

func (r *rate) getValue() float64 {
	r.Lock()
	defer r.Unlock()
	return r.value
}

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

func newRateAccumulator(prefix string, labels string) *rateAccumulator {
	a := newAccumulator(prefix + "_total_" + totalUnitSuffix + labels)
	r := newRate(a, prefix+"_rate_"+rateUnitSuffix+labels)
	return &rateAccumulator{
		a: a,
		r: r,
	}
}

func (c *rateAccumulator) WriteTo(w io.Writer) {
	c.a.WriteTo(w)
	c.r.WriteTo(w)
}

type rateAccumulatorGroup struct {
	sync.Mutex

	prefix           string
	rateAccumulators map[string]*rateAccumulator
}

func newRateAccumulatorGroup(prefix string) *rateAccumulatorGroup {
	return &rateAccumulatorGroup{
		prefix:           prefix,
		rateAccumulators: make(map[string]*rateAccumulator),
	}
}

func key(a plan.NetAddr) string {
	return fmt.Sprintf(`{peer="%s"}`, a)
}

func (g *rateAccumulatorGroup) getOrCreate(a plan.NetAddr) *rateAccumulator {
	labels := key(a)
	g.Lock()
	defer g.Unlock()
	if ra, ok := g.rateAccumulators[labels]; !ok {
		ra := newRateAccumulator(g.prefix, labels)
		g.rateAccumulators[labels] = ra
		return ra
	} else {
		return ra
	}
}

func (g *rateAccumulatorGroup) reset(a plan.NetAddr) {
	g.Lock()
	defer g.Unlock()
	for k := range g.rateAccumulators {
		delete(g.rateAccumulators, k)
	}
}

func (g *rateAccumulatorGroup) update(p time.Duration) {
	g.Lock()
	defer g.Unlock()
	for _, ra := range g.rateAccumulators {
		ra.r.update(p)
	}
}

func (g *rateAccumulatorGroup) WriteTo(w io.Writer) {
	g.Lock()
	defer g.Unlock()
	for _, ra := range g.rateAccumulators {
		ra.WriteTo(w)
	}
}

func (g *rateAccumulatorGroup) GetRates(addrs []plan.NetAddr) []float64 {
	g.Lock()
	defer g.Unlock()
	rates := make([]float64, len(addrs))
	for i, a := range addrs {
		if ra, ok := g.rateAccumulators[key(a)]; ok {
			rates[i] = ra.r.getValue()
		}
	}
	return rates
}
