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
	PeerCount int
	HostList  string
	Prog      string
	Args      []string
}

func NewProc(name string, prog string, args []string, extraEnvs Envs, peer plan.PeerID, localRank int) Proc {
	configEnvs := getConfigEnvs()
	envs := Envs{
		kb.SelfSpecEnvKey:      peer.String(),
		`CUDA_VISIBLE_DEVICES`: strconv.Itoa(localRank),
		`PYTHONUNBUFFERED`:     `1`,
	}
	return Proc{
		Name: name,
		Prog: prog,
		Args: args,
		Envs: merge(merge(configEnvs, envs), extraEnvs),
		Host: peer.Host,
		// PubAddr: pubAddr[self.Host],
	}
}

func (jc JobConfig) CreateProcs(algo kb.KungFu_AllReduceAlgo) ([]Proc, plan.PeerList, error) {
	hostSpecs, err := plan.ParseHostSpec(jc.HostList)
	if err != nil {
		return nil, nil, err
	}
	pl, err := plan.GenPeerList(jc.PeerCount, hostSpecs)
	if err != nil {
		return nil, nil, err
	}
	pubAddr := make(map[string]string)
	for _, h := range hostSpecs {
		pubAddr[h.Hostname] = h.PublicAddr
	}
	configEnvs := getConfigEnvs()
	var ps []Proc
	for i, self := range pl {
		localRank, _ := pl.LocalRank(self)
		name := fmt.Sprintf("%s.%d", self.Host, self.Port)
		envs := Envs{
			kb.PeerListEnvKey:       pl.String(),     // TODO: remove it
			`KUNGFU_TEST_SELF_RANK`: strconv.Itoa(i), // FIXME: remove it
			kb.SelfSpecEnvKey:       self.String(),
			kb.AllReduceAlgoEnvKey:  algo.String(), // FIXME: remove it
			`CUDA_VISIBLE_DEVICES`:  strconv.Itoa(localRank),
			`PYTHONUNBUFFERED`:      `1`,
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

func CreateProcs(prog string, args []string, pl plan.PeerList, algo kb.KungFu_AllReduceAlgo, disableNCCL bool) ([]Proc, error) {
	configEnvs := getConfigEnvs()
	var ps []Proc
	for i, self := range pl {
		localRank, _ := pl.LocalRank(self)
		name := fmt.Sprintf("%s.%d", self.Host, self.Port)
		envs := Envs{
			kb.PeerListEnvKey:       pl.String(),
			`KUNGFU_TEST_SELF_RANK`: strconv.Itoa(i), // FIXME: remove it
			kb.SelfSpecEnvKey:       self.String(),
			kb.AllReduceAlgoEnvKey:  algo.String(),
			`CUDA_VISIBLE_DEVICES`:  strconv.Itoa(localRank),
			`PYTHONUNBUFFERED`:      `1`,
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
