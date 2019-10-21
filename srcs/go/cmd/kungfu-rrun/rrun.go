package main

import (
	"context"
	"fmt"

	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/remote"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var f run.FlagSet

func init() { run.Init(&f) }

func main() {
	hl, err := plan.ParseHostList(f.HostList)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to parse -H: %v", err))
	}
	jc := sch.JobConfig{
		Strategy:  f.Strategy,
		HostList:  hl,
		PortRange: f.PortRange,
		Prog:      f.Prog,
		Args:      f.Args,
	}
	ps, _, err := jc.CreateProcs(f.ClusterSize)
	if err != nil {
		utils.ExitErr(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	if f.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, f.Timeout)
		defer cancel()
	}
	d, err := utils.Measure(func() error {
		_, err := runner.RemoteRunAll(ctx, f.User, ps, f.VerboseLog)
		return err
	})
	log.Infof("all %d peers finished, took %s", len(ps), d)
	if err != nil {
		utils.ExitErr(err)
	}
}
