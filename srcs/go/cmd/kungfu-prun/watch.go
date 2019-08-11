package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type peerList map[string]plan.PeerSpec

func watchRun(c *kf.ConfigClient, updated chan string, prog string, args []string, configServerAddr string) {
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
		newPeers := getNewPeers(currentPeers, cs) // FIXME: also wait termination
		log.Printf("%d new %s will be created", len(newPeers), utils.Pluralize(len(newPeers), "peer", "peers"))
		newProcs := createProcs(version, newPeers, prog, args, configServerAddr)
		for _, proc := range newProcs {
			wg.Add(1)
			go runProc(ctx, proc, &wg)
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
}

func watchConfigServer(configClient *kf.ConfigClient, newVersion chan string) {
	tk := time.NewTicker(1 * time.Second)
	defer tk.Stop()

	n := -1
	for range tk.C {
		id, version, err := configClient.GetNextVersion(n)
		if err != nil {
			log.Printf("configClient.GetLatestVersion failed: %v", err)
			continue
		}
		if id != n {
			n = id
			newVersion <- version
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

func getNewPeers(pl peerList, cs plan.ClusterSpec) []plan.PeerSpec {
	var ps []plan.PeerSpec
	for _, peer := range cs.Peers {
		if _, ok := pl[peer.NetAddr.String()]; !ok {
			ps = append(ps, peer)
		}
	}
	return ps
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

func runProc(ctx context.Context, proc sch.Proc, wg *sync.WaitGroup) {
	defer wg.Done()
	r := &runner.Runner{}
	r.SetName(proc.Name)
	r.SetLogPrefix(proc.Name)
	r.SetVerbose(true)
	if err := r.Run(ctx, proc.Cmd()); err != nil {
		log.Printf("%s finished with error: %v", proc.Name, err)
		return
	}
	log.Printf("%s finished succefully", proc.Name)
}
