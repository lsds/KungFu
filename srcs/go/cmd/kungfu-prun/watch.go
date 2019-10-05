package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	prun "github.com/lsds/KungFu/srcs/go/kungfuprun"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func watchRun(localhost string, ch chan prun.Stage, prog string, args []string) {
	log.Printf("watching config server")
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()

	var wg sync.WaitGroup
	var current plan.PeerList

	reconcileCluster := func(s prun.Stage) {
		a, b := current.Diff(s.Cluster)
		del := a.On(localhost)
		add := b.On(localhost)
		log.Printf("arrived at %s, will remove %d %s (%d locally), will add %d %s (%d locally)",
			s.Checkpoint,
			len(a), utils.Pluralize(len(a), "peer", "peers"), len(del),
			len(b), utils.Pluralize(len(b), "peer", "peers"), len(add))
		// FIXME: also wait termination
		newProcs := createProcs(s.Checkpoint, add, prog, args)
		for _, proc := range newProcs {
			wg.Add(1)
			go runProc(ctx, cancel, proc, &wg, s.Checkpoint)
		}
		current = s.Cluster
	}
	reconcileCluster(<-ch)
	go func() {
		for s := range ch {
			wg.Add(1)
			reconcileCluster(s)
			wg.Done()
		}
	}()
	if *keep {
		wg.Add(1)
	}
	wg.Wait()
	log.Printf("stop watching")
}

func createProcs(version string, pl plan.PeerList, prog string, args []string) []sch.Proc {
	envs := sch.Envs{
		kb.InitSessEnvKey: version,
	}
	var procs []sch.Proc
	for _, peer := range pl {
		localRank, _ := pl.LocalRank(peer)
		name := fmt.Sprintf("%s:%d", peer.Host, peer.Port)
		procs = append(procs, sch.NewProc(name, prog, args, envs, peer, localRank))
	}
	return procs
}

var running int32

func runProc(ctx context.Context, cancel context.CancelFunc, proc sch.Proc, wg *sync.WaitGroup, version string) {
	defer wg.Done()
	r := &runner.Runner{}
	r.SetName(proc.Name)
	r.SetLogPrefix(proc.Name + "@" + version)
	r.SetVerbose(true)
	atomic.AddInt32(&running, 1)
	err := r.Run(ctx, proc.Cmd())
	n := atomic.AddInt32(&running, -1)
	if err != nil {
		log.Printf("%s finished with error: %v, %d still running", proc.Name, err, n)
		cancel()
		return
	}
	log.Printf("%s finished succefully, %d still running", proc.Name, n)
}
