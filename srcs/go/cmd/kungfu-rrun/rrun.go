package main

import (
	"context"
	"fmt"

	"github.com/lsds/KungFu/srcs/go/job"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	runner "github.com/lsds/KungFu/srcs/go/utils/runner/remote"
)

var f run.FlagSet

func init() { run.Init(&f) }

func main() {
	hl, err := plan.ParseHostList(f.HostList)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to parse -H: %v", err))
	}
	peers, err := hl.GenPeerList(f.ClusterSize, f.PortRange)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to create peers: %v", err))
	}
	j := job.Job{
		Strategy:  f.Strategy,
		HostList:  hl,
		PortRange: f.PortRange,
		Prog:      f.Prog,
		Args:      f.Args,
		LogDir:    f.LogDir,
	}
	procs := j.CreateAllProcs(peers)
	ctx, cancel := context.WithCancel(context.Background())
	if f.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, f.Timeout)
		defer cancel()
	}
	d, err := utils.Measure(func() error {
		return runner.RemoteRunAll(ctx, f.User, procs, f.VerboseLog, f.LogDir)
	})
	log.Infof("all %d peers finished, took %s", len(procs), d)
	if err != nil {
		utils.ExitErr(err)
	}
}
