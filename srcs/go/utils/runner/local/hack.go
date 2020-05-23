package local

import (
	"context"
	"os/exec"
	"path"
	"strings"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/utils/iostream"
)

const ncclBug = `Inconsistency detected by ld.so`

func (r Runner) TryRun(ctx context.Context, cmd *exec.Cmd) error {
	for i := 1; ; i++ {
		retry, err := r.tryRun(ctx, cmd)
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
		err := cmd.Wait()
		ioDone.Wait()
		done <- err
	}()
	select {
	case <-ctx.Done():
		cmd.Process.Kill()
		<-done
		return false, ctx.Err()
	case err := <-done:
		if strings.HasPrefix(firstStderr.First, ncclBug) {
			return true, err
		}
		return false, err
	}
}
