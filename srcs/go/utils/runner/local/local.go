package local

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/lsds/KungFu/srcs/go/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/utils/iostream"
	"github.com/lsds/KungFu/srcs/go/utils/xterm"
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
	logDir        string
	logFilePrefix string
	verboseLog    bool
}

func (r *Runner) SetName(name string) {
	r.name = name
}

func (r *Runner) SetVerbose(verbose bool) {
	r.verboseLog = verbose
}

func (r *Runner) SetLogFilePrefix(prefix string) {
	r.logFilePrefix = prefix
}

func (r Runner) Run(ctx context.Context, cmd *exec.Cmd) error {
	var wg sync.WaitGroup
	if stdout, err := cmd.StdoutPipe(); err == nil {
		defer stdout.Close()
		if r.verboseLog {
			wg.Add(1)
			go func() { r.streamPipe("stdout", stdout); wg.Done() }()
		}
	} else {
		return err
	}
	if stderr, err := cmd.StderrPipe(); err == nil {
		defer stderr.Close()
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
	if len(r.logDir) > 0 {
		filename = path.Join(r.logDir, filename)
	}
	dir := path.Dir(filename)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		log.Warnf("failed to create log dir %s: %v", dir, err)
	}
	f := iostream.NewLazyFile(filename)
	return iostream.Tee(in, w, f)
}

func RunAll(ctx context.Context, ps []job.Proc, verboseLog bool) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	var fail int32
	for i, proc := range ps {
		wg.Add(1)
		go func(i int, proc job.Proc) {
			r := &Runner{
				name:          proc.Name,
				color:         basicColors[i%len(basicColors)],
				verboseLog:    verboseLog,
				logFilePrefix: strings.Replace(proc.Name, "/", "-", -1),
				logDir:        proc.LogDir,
			}
			if err := r.Run(ctx, proc.Cmd()); err != nil {
				log.Errorf("#<%s> exited with error: %v", proc.Name, err)
				atomic.AddInt32(&fail, 1)
				cancel()
			} else {
				log.Debugf("#<%s> finished successfully", proc.Name)
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
