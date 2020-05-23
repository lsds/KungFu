package local

import (
	"context"
	"os/exec"
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
	redirectors := r.defaultRedirectors()
	firstStderr := &iostream.SaveFirstdWriter{}
	firstLogs := &iostream.StdWriters{Stdout: &iostream.Null{}, Stderr: firstStderr}
	redirectors = append(redirectors, firstLogs)
	err := runWith(ctx, redirectors, cmd)
	if strings.HasPrefix(firstStderr.First, nccl.Bug) {
		return true, err
	}
	return false, err
}
