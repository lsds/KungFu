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
	Parent    plan.PeerID
	HostList  plan.HostList
	PortRange plan.PortRange
	Prog      string
	Args      []string
}

func (jc JobConfig) NewProc(name string, extraEnvs Envs, peer plan.PeerID, localRank int, checkpoint string, pl plan.PeerList) Proc {
	configEnvs := getConfigEnvs()
	envs := Envs{
		kb.SelfSpecEnvKey:      peer.String(),
		`CUDA_VISIBLE_DEVICES`: strconv.Itoa(localRank),
		kb.HostListEnvKey:      jc.HostList.String(),
		kb.PortRangeEnvKey:     jc.PortRange.String(),
		kb.ParentIDEnvKey:      jc.Parent.String(),
		kb.PeerListEnvKey:      pl.String(),
		kb.InitStepEnvKey:      checkpoint,
	}

	allEnvs := merge(merge(configEnvs, envs), extraEnvs)

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

func (jc JobConfig) CreateProcs(np int, strategy kb.Strategy) ([]Proc, plan.PeerList, error) {
	pl, err := jc.HostList.GenPeerList(np, jc.PortRange)
	if err != nil {
		return nil, nil, err
	}
	pubAddr := make(map[uint32]string)
	for _, h := range jc.HostList {
		pubAddr[h.IPv4] = h.PublicAddr
	}
	configEnvs := getConfigEnvs()
	var ps []Proc
	for i, self := range pl {
		localRank, _ := pl.LocalRank(self)
		name := fmt.Sprintf("%s.%d", plan.FormatIPv4(self.IPv4), self.Port)
		envs := Envs{
			kb.ParentIDEnvKey:          jc.Parent.String(),
			kb.PeerListEnvKey:          pl.String(),
			kb.HostListEnvKey:          jc.HostList.String(),
			kb.PortRangeEnvKey:         jc.PortRange.String(),
			`KUNGFU_TEST_SELF_RANK`:    strconv.Itoa(i), // FIXME: remove it
			kb.SelfSpecEnvKey:          self.String(),
			kb.AllReduceStrategyEnvKey: strategy.String(), // FIXME: remove it
		}
		proc := jc.NewProc(name, merge(configEnvs, envs), self, localRank, "", nil)
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
