package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func watchRun(c *kf.ConfigClient, selfIP string, updated chan string, prog string, args []string, configServerAddr string) {
	log.Printf("watching config server")
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()

	var wg sync.WaitGroup
	currentPeers := make(peerList)

	reconcileCluster := func(version string) {
		var cs plan.ClusterSpec
		if err := c.GetConfig(version, kb.ClusterSpecEnvKey, &cs); err != nil {
			log.Printf("%v", err)
			return
		}
		log.Printf("updated to %q", version)
		newPeers, removedPeers := diffPeers(currentPeers, cs) // FIXME: also wait termination
		log.Printf("%d new %s will be created, %d old %s will be removed",
			len(newPeers), utils.Pluralize(len(newPeers), "peer", "peers"),
			len(removedPeers), utils.Pluralize(len(removedPeers), "peer", "peers"))
		newProcs := createProcs(version, newPeers, prog, args, configServerAddr)
		localProcs := sch.ForHost(selfIP, newProcs)
		log.Printf("%d new %s will be created on this host", len(localProcs), utils.Pluralize(len(localProcs), "proc", "proc"))
		for _, proc := range localProcs {
			wg.Add(1)
			go runProc(ctx, cancel, proc, &wg, version)
		}
		currentPeers = makePeerList(cs)
	}
	reconcileCluster(<-updated)
	go func() {
		for version := range updated {
			wg.Add(1)
			reconcileCluster(version)
			wg.Done()
		}
	}()
	if *keep {
		wg.Add(1)
	}
	wg.Wait()
	log.Printf("stop watching")
}

func watchConfigServer(configClient *kf.ConfigClient, newVersion chan string) {
	tk := time.NewTicker(*watchPeriod)
	defer tk.Stop()
	last := "-1"
	for range tk.C {
		next, err := configClient.GetNextVersion(last)
		if err != nil {
			log.Printf("configClient.GetLatestVersion(%s) failed: %v", last, err)
			continue
		}
		if next != last {
			last = next
			newVersion <- next
		}
	}
}

func makePeerList(cs plan.ClusterSpec) peerList {
	pl := make(peerList)
	for _, peer := range cs.Peers {
		pl[peer.NetAddr.String()] = peer
	}
	return pl
}

func diffPeers(oldPeers peerList, cs plan.ClusterSpec) ([]plan.PeerSpec, []plan.PeerSpec) {
	newPeers := makePeerList(cs)
	return newPeers.Sub(oldPeers), oldPeers.Sub(newPeers)
}

func createProcs(version string, peers []plan.PeerSpec, prog string, args []string, configServerAddr string) []sch.Proc {
	envs := sch.Envs{
		kc.ConfigServerEnvKey: configServerAddr,
		kb.InitSessEnvKey:     version,
	}
	var procs []sch.Proc
	for _, peer := range peers {
		name := fmt.Sprintf("%s:%d", peer.NetAddr.Host, peer.NetAddr.Port)
		procs = append(procs, sch.NewProc(name, prog, args, envs, peer))
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
