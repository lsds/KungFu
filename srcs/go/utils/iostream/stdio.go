package iostream

import (
	"io"
	"os"
	"sync"
)

var Std = StdWriters{
	Stdout: os.Stdout,
	Stderr: os.Stderr,
}

type StdReaders struct {
	Stdout io.Reader
	Stderr io.Reader
}

type StdWriters struct {
	Stdout io.Writer
	Stderr io.Writer
}

func (r *StdReaders) Stream(ws ...*StdWriters) interface{ Wait() } {
	var outs, errs []io.Writer
	for _, w := range ws {
		outs = append(outs, w.Stdout)
		errs = append(errs, w.Stderr)
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		Tee(r.Stdout, outs...)
		wg.Done()
	}()
	go func() {
		Tee(r.Stderr, errs...)
		wg.Done()
	}()
	return &wg
}
