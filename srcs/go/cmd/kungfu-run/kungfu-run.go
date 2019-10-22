package main

import (
	"context"
	"fmt"
	"os"
	"time"

	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var f run.FlagSet

func init() { run.Init(&f) }

func main() {
	if len(f.Logfile) > 0 {
		lf, err := os.Create(f.Logfile)
		if err != nil {
			utils.ExitErr(err)
		}
		defer lf.Close()
		log.SetOutput(lf)
	}
	t0 := time.Now()
	defer func(prog string) { log.Infof("%s took %s", prog, time.Since(t0)) }(utils.ProgName())
	selfIPv4, err := run.InferSelfIPv4(f.Self, f.NIC)
	if err != nil {
		utils.ExitErr(err)
	}
	log.Infof("Using self=%s", plan.FormatIPv4(selfIPv4))
	hl, err := run.ResolveHostList(f.HostList, f.NIC)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to parse -H: %v", err))
	}
	parent := plan.PeerID{IPv4: selfIPv4, Port: uint16(f.Port)}
	parents := func() plan.PeerList {
		var ps plan.PeerList
		for _, h := range hl {
			ps = append(ps, plan.PeerID{IPv4: h.IPv4, Port: uint16(f.Port)})
		}
		return ps
	}()
	if _, ok := parents.Lookup(parent); !ok {
		utils.ExitErr(fmt.Errorf("%s not in %s", parent, parents))
	}
	peers, err := hl.GenPeerList(f.ClusterSize, f.PortRange)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to create peers: %v", err))
	}
	jc := sch.JobConfig{
		Strategy:  f.Strategy,
		Parent:    parent,
		HostList:  hl,
		PortRange: f.PortRange,
		Prog:      f.Prog,
		Args:      f.Args,
	}
	ctx, cancel := context.WithCancel(context.Background())
	if f.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, f.Timeout)
		defer cancel()
	}
	if f.Watch {
		ch := make(chan run.Stage, 1)
		ch <- run.Stage{Cluster: peers, Checkpoint: f.Checkpoint}
		watchRun(ctx, parent, parents, ch, jc)
	} else {
		simpleRun(ctx, selfIPv4, peers, jc)
	}
}

func simpleRun(ctx context.Context, selfIPv4 uint32, pl plan.PeerList, jc sch.JobConfig) {
	procs := jc.CreateProcs(pl.On(selfIPv4))
	log.Infof("will parallel run %d instances of %s with %q", len(procs), jc.Prog, jc.Args)
	d, err := utils.Measure(func() error { return runner.LocalRunAll(ctx, procs, f.VerboseLog) })
	log.Infof("all %d/%d local peers finished, took %s", len(procs), len(pl), d)
	if err != nil {
		utils.ExitErr(err)
	}
}
