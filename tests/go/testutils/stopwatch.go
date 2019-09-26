package testutils

import (
	"fmt"
	"time"
)

type StopWatch struct {
	t0 time.Time
}

func NewStopWatch() *StopWatch {
	return &StopWatch{
		t0: time.Now(),
	}
}

func (w *StopWatch) Stop(f func(time.Duration)) {
	d := time.Since(w.t0)
	if f != nil {
		f(d)
		return
	}
	fmt.Printf("took %s\n", d)
}
