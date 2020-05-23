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
	Name          string
	Color         xterm.Color
	LogDir        string
	LogFilePrefix string
	VerboseLog    bool
}

// Run a command with context
func (r Runner) Run(ctx context.Context, cmd *exec.Cmd) error {
	return runWith(ctx, r.defaultRedirectors(), cmd)
}

func (r Runner) defaultRedirectors() []*iostream.StdWriters {
	var redirectors []*iostream.StdWriters
	if r.VerboseLog {
		redirectors = append(redirectors, iostream.NewXTermRedirector(r.Name, r.Color))
	}
	if len(r.LogFilePrefix) > 0 {
		redirectors = append(redirectors, iostream.NewFileRedirector(path.Join(r.LogDir, r.LogFilePrefix)))
	}
	return redirectors
}

func runWith(ctx context.Context, redirectors []*iostream.StdWriters, cmd *exec.Cmd) error {
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

	ioDone := results.Stream(redirectors...)
	if err := cmd.Start(); err != nil {
		return err
	}
	done := make(chan error)
	go func() {
		ioDone.Wait() // call this before cmd.Wait!
		err := cmd.Wait()
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
				Name:          proc.Name,
				Color:         xterm.BasicColors.Choose(i),
				VerboseLog:    verboseLog,
				LogFilePrefix: strings.Replace(proc.Name, "/", "-", -1),
				LogDir:        proc.LogDir,
			}
			if err := r.TryRun(ctx, proc); err != nil {
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
