package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path"
	"syscall"
	"time"

	"github.com/lsds/KungFu/srcs/go/job"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var f run.FlagSet

func init() { run.Init(&f) }

func main() {
	if logfile := f.Logfile; len(logfile) > 0 {
		if len(f.LogDir) > 0 {
			logfile = path.Join(f.LogDir, logfile)
		}
		dir := path.Dir(logfile)
		if err := os.MkdirAll(dir, os.ModePerm); err != nil {
			log.Warnf("failed to create log dir %s: %v", dir, err)
		}
		lf, err := os.Create(logfile)
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
	j := job.Job{
		Strategy:  f.Strategy,
		Parent:    parent,
		HostList:  hl,
		PortRange: f.PortRange,
		Prog:      f.Prog,
		Args:      f.Args,
		LogDir:    f.LogDir,
	}
	ctx, cancel := context.WithCancel(context.Background())
	trap(cancel)
	if f.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, f.Timeout)
		defer cancel()
	}
	if f.Watch {
		ch := make(chan run.Stage, 1)
		ch <- run.Stage{Cluster: peers, Checkpoint: f.Checkpoint}
		run.WatchRun(ctx, parent, parents, ch, j)
	} else {
		run.SimpleRun(ctx, localhostIPv4, peers, j, f.VerboseLog)
	}
}

func trap(cancel context.CancelFunc) {
	c := make(chan os.Signal)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		sig := <-c
		log.Warnf("%s trapped", sig)
		cancel()
		log.Debugf("cancelled")
	}()
}
