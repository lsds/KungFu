package kungfurun

import (
	"context"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func SimpleRun(ctx context.Context, selfIPv4 uint32, pl plan.PeerList, jc sch.JobConfig, verboseLog bool) {
	procs := jc.CreateProcs(pl, selfIPv4)
	log.Infof("will parallel run %d instances of %s with %q", len(procs), jc.Prog, jc.Args)
	d, err := utils.Measure(func() error { return runner.LocalRunAll(ctx, procs, verboseLog) })
	log.Infof("all %d/%d local peers finished, took %s", len(procs), len(pl), d)
	if err != nil {
		utils.ExitErr(err)
	}
}
