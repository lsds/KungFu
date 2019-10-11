package main

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func watchRun(ctx context.Context, parent plan.PeerID, parents plan.PeerList, ch chan run.Stage, jc sch.JobConfig) {
	ctx, cancel := context.WithCancel(ctx)
	globalCtx, globalCancel := context.WithCancel(ctx)
	server, err := rch.NewServer(run.NewHandler(parent, ch, globalCancel))
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to create server: %v", err))
	}
	go server.Serve()
	defer server.Close()
	log.Infof("watching config server")

	var all sync.WaitGroup
	var current plan.PeerList
	var running int32
	gs := make(map[plan.PeerID]*sync.WaitGroup)

	reconcileCluster := func(s run.Stage) {
		a, b := current.Diff(s.Cluster)
		del := a.On(parent.Host)
		add := b.On(parent.Host)
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
			go func(g *sync.WaitGroup, id plan.PeerID, s run.Stage) {
				localRank, _ := s.Cluster.LocalRank(id)
				name := fmt.Sprintf("%s.%d", id.Host, id.Port)
				envs := sch.Envs{
					kb.InitSessEnvKey:   s.Checkpoint,
					kb.CheckpointEnvKey: s.Checkpoint,
				}
				proc := jc.NewProc(name, envs, id, localRank, s.Checkpoint, s.Cluster)
				atomic.AddInt32(&running, 1)
				runProc(ctx, cancel, proc, s.Checkpoint)
				n := atomic.AddInt32(&running, -1)
				log.Debugf("%d peer is still running on this host", n)
				g.Done()
				all.Done()
			}(gs[id], id, s)
			log.Debugf("peer %d/%d created", i, len(add))
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

	if hostRank, _ := parents.Lookup(parent); len(parents) > 1 {
		if hostRank > 0 {
			<-globalCtx.Done()
			err := globalCtx.Err()
			log.Infof("global context is done: %v", err)
			all.Wait()
		} else {
			all.Wait()
			router := rch.NewRouter(parent, nil)
			for _, p := range parents[1:] {
				router.Send(p.WithName("exit"), nil, rch.ConnControl, 0)
			}
		}
	} else {
		all.Wait()
	}
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
