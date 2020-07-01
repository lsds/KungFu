package main

import (
	"context"
	"os"

	"github.com/lsds/KungFu/srcs/go/kungfu/job"
	"github.com/lsds/KungFu/srcs/go/kungfu/runner"
	"github.com/lsds/KungFu/srcs/go/kungfu/runtime"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/runner/remote"
)

var f runner.FlagSet

func init() { runner.Init(&f, os.Args) }

func main() {
	j := job.Job{
		Strategy:  f.Strategy,
		HostList:  f.HostList,
		PortRange: f.PortRange,
		Prog:      f.Prog,
		Args:      f.Args,
		LogDir:    f.LogDir,
	}
	ctx, cancel := context.WithCancel(context.Background())
	if f.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, f.Timeout)
		defer cancel()
	}
	sp := runtime.SystemParameters{
		User:            f.User,
		WorkerPortRange: f.PortRange,
		RunnerPort:      uint16(f.Port),
		HostList:        f.HostList,
		ClusterSize:     f.ClusterSize,
		Nic:             f.NIC,
	}
	if err := remote.RunStaticKungFuJob(ctx, j, sp, f.Quiet); err != nil {
		utils.ExitErr(err)
	}
}
