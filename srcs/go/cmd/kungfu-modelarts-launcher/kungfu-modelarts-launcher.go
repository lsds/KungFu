package main

import (
	"context"
	"time"

	"github.com/lsds/KungFu/srcs/go/job"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/platforms/modelarts"
	"github.com/lsds/KungFu/srcs/go/utils"
	runner "github.com/lsds/KungFu/srcs/go/utils/runner/local"
)

var f run.FlagSet

func init() { run.Init(&f) }

func main() {
	t0 := time.Now()
	defer func(prog string) { log.Infof("%s took %s", prog, time.Since(t0)) }(utils.ProgName())
	env, err := modelarts.ParseEnv()
	if err != nil {
		utils.ExitErr(err)
	}
	j := job.Job{
		Strategy:  f.Strategy,
		PortRange: f.PortRange,
		Prog:      f.Prog,
		Args:      f.Args,
		LogDir:    f.LogDir,
	}
	procs := []job.Proc{
		j.NewProc(env.Self, 0, "", env.PeerList),
	}
	ctx, cancel := context.WithCancel(context.Background())
	if f.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, f.Timeout)
		defer cancel()
	}
	if err := runner.RunAll(ctx, procs, f.VerboseLog); err != nil {
		utils.ExitErr(err)
	}
}
