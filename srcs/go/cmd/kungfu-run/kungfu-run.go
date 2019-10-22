package main

import (
	"context"
	"fmt"
	"os"
	"time"

	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
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
	localhostIPv4, err := run.InferSelfIPv4(f.Self, f.NIC)
	if err != nil {
		utils.ExitErr(err)
	}
	log.Infof("Using self=%s", plan.FormatIPv4(localhostIPv4))
	parent := plan.PeerID{IPv4: localhostIPv4, Port: uint16(f.Port)}
	var parents plan.PeerList
	var hl plan.HostList
	var peers plan.PeerList
	if len(f.HostList) > 0 {
		hl, err = run.ResolveHostList(f.HostList, f.NIC)
		if err != nil {
			utils.ExitErr(fmt.Errorf("failed to parse -H: %v", err))
		}
		for _, h := range hl {
			parents = append(parents, plan.PeerID{IPv4: h.IPv4, Port: uint16(f.Port)})
		}
		if _, ok := parents.Lookup(parent); !ok {
			utils.ExitErr(fmt.Errorf("%s not in %s", parent, parents))
		}
		peers, err = hl.GenPeerList(f.ClusterSize, f.PortRange)
		if err != nil {
			utils.ExitErr(fmt.Errorf("failed to create peers: %v", err))
		}
	} else {
		peers, err = run.ResolvePeerList(localhostIPv4, uint16(f.Port), f.PeerList)
		if err != nil {
			utils.ExitErr(fmt.Errorf("failed to resolve peers: %v", err))
		}
		log.Infof("-P resolved as %s", peers)
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
		run.WatchRun(ctx, parent, parents, ch, jc)
	} else {
		run.SimpleRun(ctx, localhostIPv4, peers, jc, f.VerboseLog)
	}
}
