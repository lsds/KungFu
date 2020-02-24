package utils

import (
	"fmt"
	"os"
	"time"
)

type stallDetector struct {
	name string
	tk   *time.Ticker
}

func InstallStallDetector(name string) *stallDetector {
	s := &stallDetector{
		name: name,
		tk:   time.NewTicker(3 * time.Second),
	}
	go s.start()
	return s
}

func (s *stallDetector) start() {
	t0 := time.Now()
	var hasStalled bool
	for range s.tk.C {
		hasStalled = true
		fmt.Fprintf(os.Stderr, "%s stalled\n", s.name)
	}
	if hasStalled {
		fmt.Fprintf(os.Stderr, "%s recovered after %s\n", s.name, time.Since(t0))
	}
}

func (s *stallDetector) Stop() {
	s.tk.Stop()
}
