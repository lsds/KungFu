package local

import (
	"context"
	"fmt"
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

// Run a command with context
func (r Runner) Run(ctx context.Context, cmd *exec.Cmd) error {
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	defer stdout.Close()
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	defer stderr.Close()
	results := iostream.StdReaders{Stdout: stdout, Stderr: stderr}
	var redirectors []*iostream.StdWriters
	if r.verboseLog {
		redirectors = append(redirectors, iostream.NewXTermRedirector(r.name, r.color))
	}
	if len(r.logFilePrefix) > 0 {
		redirectors = append(redirectors, iostream.NewFileRedirector(path.Join(r.logDir, r.logFilePrefix)))
	}
	ioDone := results.Stream(redirectors...)
	if err := cmd.Start(); err != nil {
		return err
	}
	done := make(chan error)
	go func() {
		err := cmd.Wait()
		ioDone.Wait()
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
				color:         xterm.BasicColors.Choose(i),
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
		return fmt.Errorf("%d tasks failed", fail)
	}
	return nil
}
