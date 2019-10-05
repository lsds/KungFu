package main

import (
	"context"
	"fmt"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	prun "github.com/lsds/KungFu/srcs/go/kungfuprun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func watchRun(localhost string, ch chan prun.Stage, jc sch.JobConfig) {
	log.Infof("watching config server")
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()

	var all sync.WaitGroup
	var current plan.PeerList
	gs := make(map[plan.PeerID]*sync.WaitGroup)

	reconcileCluster := func(s prun.Stage) {
		a, b := current.Diff(s.Cluster)
		del := a.On(localhost)
		add := b.On(localhost)
		log.Infof("arrived at %q, np=%d, will remove %d %s (%d locally), will add %d %s (%d locally)",
			s.Checkpoint, len(s.Cluster),
			len(a), utils.Pluralize(len(a), "peer", "peers"), len(del),
			len(b), utils.Pluralize(len(b), "peer", "peers"), len(add))
		log.Debugf("waiting %d peers to stop", len(del))
		for _, id := range del {
			gs[id].Wait()
			delete(gs, id)
		}
		log.Debugf("%d peers removed", len(del))
		for i, id := range add {
			gs[id] = new(sync.WaitGroup)
			gs[id].Add(1)
			all.Add(1)
			go func(g *sync.WaitGroup, id plan.PeerID, s prun.Stage) {
				localRank, _ := s.Cluster.LocalRank(id)
				name := fmt.Sprintf("%s.%d", id.Host, id.Port)
				envs := sch.Envs{
					kb.InitSessEnvKey:   s.Checkpoint,
					kb.CheckpointEnvKey: s.Checkpoint,
				}
				proc := jc.NewProc(name, envs, id, localRank, s.Checkpoint, s.Cluster)
				runProc(ctx, cancel, proc, s.Checkpoint)
				g.Done()
				all.Done()
			}(gs[id], id, s)
			log.Debugf("peers %d/%d created", i, len(add))
		}
		log.Debugf("%d peers created", len(add))
		current = s.Cluster
	}
	reconcileCluster(<-ch)
	go func() {
		for s := range ch {
			all.Add(1)
			reconcileCluster(s)
			all.Done()
		}
	}()
	if *keep {
		err := <-ctx.Done()
		log.Infof("context is done: %v", err)
	}
	all.Wait()
	log.Infof("stop watching")
}

func runProc(ctx context.Context, cancel context.CancelFunc, proc sch.Proc, version string) {
	r := &runner.Runner{}
	r.SetName(proc.Name)
	r.SetLogPrefix(proc.Name + "@" + version)
	r.SetVerbose(true)
	err := r.Run(ctx, proc.Cmd())
	if err != nil {
		log.Infof("%s finished with error: %v", proc.Name, err)
		cancel()
		return
	}
}
