package local

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/lsds/KungFu/srcs/go/iostream"
	"github.com/lsds/KungFu/srcs/go/log"
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

func (r *Runner) SetName(name string) {
	r.name = name
}

func (r *Runner) SetVerbose(verbose bool) {
	r.verboseLog = verbose
}

func (r *Runner) SetLogPrefix(prefix string) {
	r.logFilePrefix = prefix
}

func (r Runner) Run(ctx context.Context, cmd *exec.Cmd) error {
	var wg sync.WaitGroup
	if stdout, err := cmd.StdoutPipe(); err == nil {
		if r.verboseLog {
			wg.Add(1)
			go func() { r.streamPipe("stdout", stdout); wg.Done() }()
		}
	} else {
		return err
	}
	if stderr, err := cmd.StderrPipe(); err == nil {
		if r.verboseLog {
			wg.Add(1)
			go func() { r.streamPipe("stderr", stderr); wg.Done() }()
		}
	} else {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	done := make(chan error)
	go func() {
		err := cmd.Wait()
		wg.Wait()
		done <- err
	}()
	select {
	case <-ctx.Done():
		cmd.Process.Kill()
		<-done
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
	rName := r.name
	if r.color != nil {
		rName = r.color.S(rName)
	}
	w := &XtermWriter{Prefix: rName + "::" + name}
	filename := name + ".log"
	if len(r.logFilePrefix) > 0 {
		filename = r.logFilePrefix + "-" + filename
	}
	f, err := os.Create(filename)
	if err != nil {
		log.Errorf("failed to create log file: %v", err)
		return iostream.Tee(in, w)
	}
	return iostream.Tee(in, w, f)
}

func LocalRunAll(ctx context.Context, ps []sch.Proc, verboseLog bool) error {
	ctx, cancel := context.WithCancel(ctx)
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
			if err := r.Run(ctx, proc.Cmd()); err != nil {
				log.Errorf("%s #%s exited with error: %v", xterm.Red.S("[E]"), proc.Name, err)
				atomic.AddInt32(&fail, 1)
				cancel()
			} else {
				log.Infof("%s #%s finished successfully", xterm.Green.S("[I]"), proc.Name)
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
