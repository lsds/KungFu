package scheduler

import (
	"fmt"
	"os"
	"strconv"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type JobConfig struct {
	Strategy  kb.Strategy
	Parent    plan.PeerID
	HostList  plan.HostList
	PortRange plan.PortRange
	Prog      string
	Args      []string
}

func (jc JobConfig) NewProc(name string, peer plan.PeerID, localRank int, checkpoint string, pl plan.PeerList) Proc {
	envs := Envs{
		kb.SelfSpecEnvKey:          peer.String(),
		`CUDA_VISIBLE_DEVICES`:     strconv.Itoa(localRank),
		kb.HostListEnvKey:          jc.HostList.String(),
		kb.PortRangeEnvKey:         jc.PortRange.String(),
		kb.ParentIDEnvKey:          jc.Parent.String(),
		kb.PeerListEnvKey:          pl.String(),
		kb.CheckpointEnvKey:        checkpoint,
		kb.AllReduceStrategyEnvKey: jc.Strategy.String(),
	}
	allEnvs := merge(getConfigEnvs(), envs)
	allEnvs.addIfMissing(`DYLD_LIBRARY_PATH`, run.DefaultLdLibraryPath)
	allEnvs.addIfMissing(`PYTHONUNBUFFERED`, `1`)
	var pubAddr string
	for _, h := range jc.HostList {
		if h.IPv4 == peer.IPv4 {
			pubAddr = h.PublicAddr
		}
	}

	return Proc{
		Name:    name,
		Prog:    jc.Prog,
		Args:    jc.Args,
		Envs:    allEnvs,
		IPv4:    peer.IPv4,
		PubAddr: pubAddr,
	}
}

func (jc JobConfig) CreateProcs(np int) ([]Proc, plan.PeerList, error) {
	pl, err := jc.HostList.GenPeerList(np, jc.PortRange)
	if err != nil {
		return nil, nil, err
	}
	var ps []Proc
	for _, self := range pl {
		localRank, _ := pl.LocalRank(self)
		name := fmt.Sprintf("%s.%d", plan.FormatIPv4(self.IPv4), self.Port)
		proc := jc.NewProc(name, self, localRank, "", pl)
		ps = append(ps, proc)
	}
	return ps, pl, nil
}

func ForHost(myHost uint32, ps []Proc) []Proc {
	var myPs []Proc
	for _, p := range ps {
		if p.IPv4 == myHost {
			myPs = append(myPs, p)
		}
	}
	return myPs
}

func getConfigEnvs() Envs {
	envs := make(Envs)
	for _, k := range kc.ConfigEnvKeys {
		if val := os.Getenv(k); len(val) > 0 {
			envs[k] = val
		}
	}
	return envs
}
