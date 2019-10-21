package main

import (
	"context"
	"fmt"
	"time"

	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/platforms/modelarts"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var f run.FlagSet

func init() { run.Init(&f) }

func name(id plan.PeerID) string {
	return fmt.Sprintf("%s.%d", plan.FormatIPv4(id.IPv4), id.Port)
}

func main() {
	t0 := time.Now()
	defer func(prog string) { log.Infof("%s took %s", prog, time.Since(t0)) }(utils.ProgName())
	env, err := modelarts.ParseEnv()
	if err != nil {
		utils.ExitErr(err)
	}
	jc := sch.JobConfig{
		HostList:  plan.HostList{{}}, // FIXME: make sure it's not used
		Strategy:  f.Strategy,
		PortRange: f.PortRange,
		Prog:      f.Prog,
		Args:      f.Args,
	}
	procs := []sch.Proc{
		jc.NewProc(name(env.Self), env.Self, 0, "", env.PeerList),
	}
	ctx, cancel := context.WithCancel(context.Background())
	if f.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, f.Timeout)
		defer cancel()
	}
	if err := runner.LocalRunAll(ctx, procs, f.VerboseLog); err != nil {
		utils.ExitErr(err)
	}
}
