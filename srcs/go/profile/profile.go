package profile

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"time"
)

var (
	now   = time.Now
	since = time.Since
)

var Default = New()

type profiler struct {
	sync.Mutex
	counts         map[string]int64
	minDurations   map[string]time.Duration
	maxDurations   map[string]time.Duration
	totalDurations map[string]time.Duration

	events []event
}

type scope struct {
	name     string
	begin    time.Time
	profiler *profiler
}

type event struct {
	Name    string
	BeginNs int64
	EndNs   int64
}

func New() *profiler {
	return &profiler{
		counts:         make(map[string]int64),
		minDurations:   make(map[string]time.Duration),
		maxDurations:   make(map[string]time.Duration),
		totalDurations: make(map[string]time.Duration),
	}
}

func (p *profiler) Profile(name string) *scope {
	return &scope{
		name:     name,
		begin:    now(),
		profiler: p,
	}
}

func (p *profiler) add(name string, d time.Duration) {
	p.Lock()
	defer p.Unlock()
	p.counts[name]++
	p.totalDurations[name] += d
	if val, ok := p.minDurations[name]; !ok || d < val {
		p.minDurations[name] = d
	}
	if val, ok := p.maxDurations[name]; !ok || d > val {
		p.maxDurations[name] = d
	}
}

func (p *profiler) logEvent(name string, begin, end time.Time) {
	p.Lock()
	defer p.Unlock()
	p.events = append(p.events, event{Name: name, BeginNs: begin.UnixNano(), EndNs: end.UnixNano()})
}

func (p *profiler) WriteEvents(w io.Writer) {
	p.Lock()
	defer p.Unlock()
	for _, e := range p.events {
		fmt.Fprintf(w, "%d %d %s\n", e.BeginNs, e.EndNs, e.Name)
	}
}

func (p *profiler) WriteSummary(w io.Writer) {
	p.Lock()
	defer p.Unlock()
	var names []string
	for name := range p.counts {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool { return p.totalDurations[names[i]] > p.totalDurations[names[j]] })

	type record struct {
		count int64
		min   time.Duration
		max   time.Duration
		total time.Duration
		name  string

		mean  time.Duration
		speed float64
	}

	var records []record
	for _, name := range names {
		cnt := p.counts[name]
		tot := p.totalDurations[name]
		mean := tot / time.Duration(cnt)
		records = append(records, record{
			name:  name,
			min:   p.minDurations[name],
			max:   p.maxDurations[name],
			total: tot,
			count: cnt,
			mean:  mean,
		})
	}

	th := []string{"count", "mean", "min", "max", "total", "call site"}
	var rows [][]string
	for _, r := range records {
		rows = append(rows, []string{
			fmt.Sprintf("%d", r.count),
			fmt.Sprintf("%s", r.mean),
			fmt.Sprintf("%s", r.min),
			fmt.Sprintf("%s", r.max),
			fmt.Sprintf("%s", r.total),
			fmt.Sprintf("%s", r.name),
		})
	}
	showTable(w, th, rows)
}

func (s *scope) Done() {
	end := now()
	d := end.Sub(s.begin)
	s.profiler.add(s.name, d)
	s.profiler.logEvent(s.name, s.begin, end)
}
