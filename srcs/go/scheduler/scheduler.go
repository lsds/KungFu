package scheduler

import (
	"fmt"
	"os"
	"strconv"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
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
		`PYTHONUNBUFFERED`:     `1`,
		kb.HostListEnvKey:      jc.HostList.String(),
		kb.PortRangeEnvKey:     jc.PortRange.String(),
		kb.ParentIDEnvKey:      jc.Parent.String(),
		kb.PeerListEnvKey:      pl.String(),
		kb.InitStepEnvKey:      checkpoint,
	}
	return Proc{
		Name: name,
		Prog: jc.Prog,
		Args: jc.Args,
		Envs: merge(merge(configEnvs, envs), extraEnvs),
		Host: peer.Host,
		// PubAddr: pubAddr[self.Host],
	}
}

func (jc JobConfig) CreateProcs(np int, strategy kb.Strategy) ([]Proc, plan.PeerList, error) {
	pl, err := jc.HostList.GenPeerList(np, jc.PortRange)
	if err != nil {
		return nil, nil, err
	}
	pubAddr := make(map[string]string)
	for _, h := range jc.HostList {
		pubAddr[h.Hostname] = h.PublicAddr
	}
	configEnvs := getConfigEnvs()
	var ps []Proc
	for i, self := range pl {
		localRank, _ := pl.LocalRank(self)
		name := fmt.Sprintf("%s.%d", self.Host, self.Port)
		envs := Envs{
			kb.ParentIDEnvKey:          jc.Parent.String(),
			kb.PeerListEnvKey:          pl.String(),
			kb.HostListEnvKey:          jc.HostList.String(),
			kb.PortRangeEnvKey:         jc.PortRange.String(),
			`KUNGFU_TEST_SELF_RANK`:    strconv.Itoa(i), // FIXME: remove it
			kb.SelfSpecEnvKey:          self.String(),
			kb.AllReduceStrategyEnvKey: strategy.String(), // FIXME: remove it
			`CUDA_VISIBLE_DEVICES`:     strconv.Itoa(localRank),
			`PYTHONUNBUFFERED`:         `1`,
		}
		ps = append(ps, Proc{
			Name:    name,
			Prog:    jc.Prog,
			Args:    jc.Args,
			Envs:    merge(configEnvs, envs),
			Host:    self.Host,
			PubAddr: pubAddr[self.Host],
		})
	}
	return ps, pl, nil
}

func CreateProcs(prog string, args []string, pl plan.PeerList, strategy kb.Strategy, disableNCCL bool) ([]Proc, error) {
	configEnvs := getConfigEnvs()
	var ps []Proc
	for i, self := range pl {
		localRank, _ := pl.LocalRank(self)
		name := fmt.Sprintf("%s.%d", self.Host, self.Port)
		envs := Envs{
			kb.PeerListEnvKey:          pl.String(),
			`KUNGFU_TEST_SELF_RANK`:    strconv.Itoa(i), // FIXME: remove it
			kb.SelfSpecEnvKey:          self.String(),
			kb.AllReduceStrategyEnvKey: strategy.String(),
			`CUDA_VISIBLE_DEVICES`:     strconv.Itoa(localRank),
			`PYTHONUNBUFFERED`:         `1`,
		}
		if disableNCCL {
			envs[`KUNGFU_DISABLE_NCCL`] = `1`
		}
		ps = append(ps, Proc{
			Name:    name,
			Prog:    prog,
			Args:    args,
			Envs:    merge(configEnvs, envs),
			Host:    self.Host,
			PubAddr: self.Host,
		})
	}
	return ps, nil
}

func ForHost(myHost string, ps []Proc) []Proc {
	var myPs []Proc
	for _, p := range ps {
		if p.Host == myHost {
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
