package runner

import (
	"context"

	"github.com/lsds/KungFu/srcs/go/kungfu/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/runner/local"
)

func SimpleRun(ctx context.Context, selfIPv4 uint32, cluster plan.Cluster, j job.Job, verboseLog bool) {
	procs := j.CreateProcs(cluster, selfIPv4)
	log.Infof("will parallel run %d instances of %s with %q", len(procs), j.Prog, j.Args)
	d, err := utils.Measure(func() error { return local.RunAll(ctx, procs, verboseLog) })
	log.Infof("all %d/%d local peers finished, took %s", len(procs), len(cluster.Workers), d)
	if err != nil {
		utils.ExitErr(err)
	}
}
