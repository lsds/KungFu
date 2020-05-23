package local

import (
	"context"
	"os/exec"
	"path"
	"strings"

	"github.com/lsds/KungFu/srcs/go/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/nccl"
	"github.com/lsds/KungFu/srcs/go/utils/iostream"
)

func (r Runner) TryRun(ctx context.Context, proc job.Proc) error {
	for i := 1; ; i++ {
		retry, err := r.tryRun(ctx, proc.Cmd())
		if err != nil && retry {
			log.Errorf("restarting for the %d-th time because of %v", i, err)
			continue
		}
		return err
	}
}

func (r Runner) tryRun(ctx context.Context, cmd *exec.Cmd) (bool, error) {
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return false, err
	}
	defer stdout.Close()
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return false, err
	}
	defer stderr.Close()
	results := iostream.StdReaders{Stdout: stdout, Stderr: stderr}
	var redirectors []*iostream.StdWriters
	if r.VerboseLog {
		redirectors = append(redirectors, iostream.NewXTermRedirector(r.Name, r.Color))
	}
	if len(r.LogFilePrefix) > 0 {
		redirectors = append(redirectors, iostream.NewFileRedirector(path.Join(r.LogDir, r.LogFilePrefix)))
	}
	firstStderr := &iostream.SaveFirstdWriter{}
	firstLogs := &iostream.StdWriters{Stdout: &iostream.Null{}, Stderr: firstStderr}
	redirectors = append(redirectors, firstLogs)
	ioDone := results.Stream(redirectors...)
	if err := cmd.Start(); err != nil {
		return false, err
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
		return false, ctx.Err()
	case err := <-done:
		if strings.HasPrefix(firstStderr.First, nccl.Bug) {
			return true, err
		}
		return false, err
	}
}
