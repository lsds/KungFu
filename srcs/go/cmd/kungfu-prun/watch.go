package main

import (
	"context"
	"fmt"
	"log"
	"sync"

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

	var all sync.WaitGroup
	var current plan.PeerList
	gs := make(map[plan.PeerID]*sync.WaitGroup)

	reconcileCluster := func(s prun.Stage) {
		a, b := current.Diff(s.Cluster)
		del := a.On(localhost)
		add := b.On(localhost)
		log.Printf("arrived at %s, will remove %d %s (%d locally), will add %d %s (%d locally)",
			s.Checkpoint,
			len(a), utils.Pluralize(len(a), "peer", "peers"), len(del),
			len(b), utils.Pluralize(len(b), "peer", "peers"), len(add))
		for _, id := range del {
			gs[id].Wait()
			delete(gs, id)
		}
		// log.Debugf("%d peers removed", len(del))
		for _, id := range add {
			gs[id] = new(sync.WaitGroup)
			gs[id].Add(1)
			all.Add(1)
			go func(g *sync.WaitGroup, id plan.PeerID, chpt string) {
				localRank, _ := s.Cluster.LocalRank(id)
				name := fmt.Sprintf("%s.%d", id.Host, id.Port)
				envs := sch.Envs{
					kb.InitSessEnvKey: s.Checkpoint,
				}
				proc := sch.NewProc(name, prog, args, envs, id, localRank)
				runProc(ctx, cancel, proc, s.Checkpoint)
				g.Done()
				all.Done()
			}(gs[id], id, s.Checkpoint)
		}
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
		log.Printf("context is done: %v", err)
	}
	all.Wait()
	log.Printf("stop watching")
}

func runProc(ctx context.Context, cancel context.CancelFunc, proc sch.Proc, version string) {
	r := &runner.Runner{}
	r.SetName(proc.Name)
	r.SetLogPrefix(proc.Name + "@" + version)
	r.SetVerbose(true)
	err := r.Run(ctx, proc.Cmd())
	if err != nil {
		log.Printf("%s finished with error: %v", proc.Name, err)
		cancel()
		return
	}
}
