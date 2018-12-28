package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luomai/kungfu/srcs/go/iostream"
	"github.com/luomai/kungfu/srcs/go/xterm"
)

var (
	basicColors = []xterm.Color{
		xterm.Green,
		xterm.Yellow,
	}

	warn = xterm.Red
)

type Proc struct {
	name string
	cmd  *exec.Cmd
}

type Runner struct {
	name       string
	color      xterm.Color
	verboseLog bool
}

func (r Runner) run(ctx context.Context, cmd *exec.Cmd) error {
	if stdout, err := cmd.StdoutPipe(); err == nil {
		if r.verboseLog {
			go r.streamPipe("stdout", stdout)
		}
	} else {
		return err
	}
	if stderr, err := cmd.StderrPipe(); err == nil {
		if r.verboseLog {
			go r.streamPipe("stderr", stderr)
		}
	} else {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	done := make(chan error)
	go func() { done <- cmd.Wait() }()
	select {
	case <-ctx.Done():
		cmd.Process.Kill()
		return ctx.Err()
	case err := <-done:
		return err
	}
}

type XtermWriter struct {
	Prefix string
}

func (x XtermWriter) Write(bs []byte) (int, error) {
	fmt.Fprintf(os.Stderr, "[%s] %s", x.Prefix, string(bs))
	return len(bs), nil
}

func (r Runner) streamPipe(name string, in io.Reader) error {
	w := &XtermWriter{Prefix: r.color.S(r.name) + "::" + name}
	// TODO: make sure r.name is a valid filename
	filename := fmt.Sprintf("%s-%s.log", r.name, name)
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	return iostream.Tee(in, w, f)
}

func runAll(ctx context.Context, ps []*Proc, verboseLog bool) error {
	t0 := time.Now()
	var wg sync.WaitGroup
	var fail int32
	for i, proc := range ps {
		wg.Add(1)
		go func(i int, proc *Proc) {
			r := &Runner{
				name:       proc.name,
				color:      basicColors[i%len(basicColors)],
				verboseLog: verboseLog,
			}
			if err := r.run(ctx, proc.cmd); err != nil {
				log.Printf("%s $%s exited with error: %v", xterm.Red.S("[E]"), proc.name, err)
				atomic.AddInt32(&fail, 1)
			} else {
				log.Printf("%s $%s finished successfully", xterm.Green.S("[I]"), proc.name)
			}
			wg.Done()
		}(i, proc)
	}
	wg.Wait()
	d := time.Since(t0)
	log.Printf("all %d tasks finished, took %s", len(ps), d)
	if fail != 0 {
		return fmt.Errorf("%d tasks failed", fail)
	}
	return nil
}
