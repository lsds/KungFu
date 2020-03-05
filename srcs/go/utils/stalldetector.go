package utils

import (
	"fmt"
	"os"
	"time"
)

type stallDetector struct {
	name    string
	tk      *time.Ticker
	stopped chan struct{}
}

func InstallStallDetector(name string) *stallDetector {
	s := &stallDetector{
		name:    name,
		tk:      time.NewTicker(3 * time.Second),
		stopped: make(chan struct{}),
	}
	go s.start()
	return s
}

func (s *stallDetector) start() {
	t0 := time.Now()
	var hasStalled bool
	for {
		select {
		case <-s.tk.C:
			hasStalled = true
			fmt.Fprintf(os.Stderr, "%s stalled for %s\n", s.name, time.Since(t0))
		case <-s.stopped:
			goto Stopped
		}
	}
Stopped:
	if hasStalled {
		fmt.Fprintf(os.Stderr, "%s recovered after %s\n", s.name, time.Since(t0))
	}
}

func (s *stallDetector) Stop() {
	s.tk.Stop()
	s.stopped <- struct{}{}
}
