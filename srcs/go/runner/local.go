package runner

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/lsds/KungFu/srcs/go/iostream"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/xterm"
)

var (
	basicColors = []xterm.Color{
		xterm.Green,
		xterm.Blue,
		xterm.Yellow,
		xterm.LightBlue,
	}

	warn = xterm.Red
)

type Runner struct {
	name          string
	color         xterm.Color
	logFilePrefix string
	verboseLog    bool
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
	filename := r.logFilePrefix + "-" + name + ".log"
	f, err := os.Create(filename)
	if err != nil {
		log.Printf("failed to create log file: %v", err)
		return iostream.Tee(in, w)
	}
	return iostream.Tee(in, w, f)
}

func LocalRunAll(ctx context.Context, ps []sch.Proc, verboseLog bool) error {
	var wg sync.WaitGroup
	var fail int32
	for i, proc := range ps {
		wg.Add(1)
		go func(i int, proc sch.Proc) {
			r := &Runner{
				name:          proc.Name,
				color:         basicColors[i%len(basicColors)],
				verboseLog:    verboseLog,
				logFilePrefix: strings.Replace(proc.Name, "/", "-", -1),
			}
			if err := r.run(ctx, proc.Cmd()); err != nil {
				log.Printf("%s #%s exited with error: %v", xterm.Red.S("[E]"), proc.Name, err)
				atomic.AddInt32(&fail, 1)
			} else {
				log.Printf("%s #%s finished successfully", xterm.Green.S("[I]"), proc.Name)
			}
			wg.Done()
		}(i, proc)
	}
	wg.Wait()
	if fail != 0 {
		return fmt.Errorf("%d peers failed", fail)
	}
	return nil
}
