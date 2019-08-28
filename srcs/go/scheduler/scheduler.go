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

func NewProc(name string, prog string, args []string, extraEnvs Envs, peer plan.PeerSpec) Proc {
	configEnvs := getConfigEnvs()
	envs := Envs{
		kb.SelfSpecEnvKey:      peer.String(),
		`CUDA_VISIBLE_DEVICES`: strconv.Itoa(peer.DeviceID),
		`PYTHONUNBUFFERED`:     `1`,
	}
	return Proc{
		Name: name,
		Prog: prog,
		Args: args,
		Envs: merge(merge(configEnvs, envs), extraEnvs),
		Host: peer.NetAddr.Host,
		// PubAddr: pubAddr[self.NetAddr.Host],
	}
}

func (jc JobConfig) CreateProcs(algo kb.KungFu_AllReduceAlgo, configServerAddr string) ([]Proc, *plan.ClusterSpec, error) {
	hostSpecs, err := plan.ParseHostSpec(jc.HostList)
	if err != nil {
		return nil, nil, err
	}
	cs, err := plan.GenClusterSpec(jc.PeerCount, hostSpecs)
	if err != nil {
		return nil, nil, err
	}
	pubAddr := make(map[string]string)
	for _, h := range hostSpecs {
		pubAddr[h.Hostname] = h.PublicAddr
	}
	configEnvs := getConfigEnvs()
	var ps []Proc
	for i, self := range cs.Peers {
		name := fmt.Sprintf("%02s/%02d/%02d-of-%02d", self.NetAddr.Host, self.DeviceID, i, len(cs.Peers))
		envs := Envs{
			kb.ClusterSpecEnvKey:    cs.String(),     // TODO: remove it
			`KUNGFU_TEST_SELF_RANK`: strconv.Itoa(i), // FIXME: remove it
			kb.SelfSpecEnvKey:       self.String(),
			kb.AllReduceAlgoEnvKey:  algo.String(), // FIXME: remove it
			`CUDA_VISIBLE_DEVICES`:  strconv.Itoa(self.DeviceID),
			`PYTHONUNBUFFERED`:      `1`,
			// TODO: add LD_PRELOAD to tcmalloc path
			// `LD_PRELOAD`:``,
		}
		if len(configServerAddr) > 0 {
			envs[kc.ConfigServerEnvKey] = configServerAddr
		}
		ps = append(ps, Proc{
			Name:    name,
			Prog:    jc.Prog,
			Args:    jc.Args,
			Envs:    merge(configEnvs, envs),
			Host:    self.NetAddr.Host,
			PubAddr: pubAddr[self.NetAddr.Host],
		})
	}
	return ps, cs, nil
}

func CreateProcs(prog string, args []string, cs *plan.ClusterSpec, algo kb.KungFu_AllReduceAlgo, disableNCCL bool) ([]Proc, error) {
	configEnvs := getConfigEnvs()
	var ps []Proc
	for i, self := range cs.Peers {
		name := fmt.Sprintf("%02s/%02d/%02d-of-%02d", self.NetAddr.Host, self.DeviceID, i, len(cs.Peers))
		envs := Envs{
			kb.ClusterSpecEnvKey:    cs.String(),
			`KUNGFU_TEST_SELF_RANK`: strconv.Itoa(i), // FIXME: remove it
			kb.SelfSpecEnvKey:       self.String(),
			kb.AllReduceAlgoEnvKey:  algo.String(),
			`CUDA_VISIBLE_DEVICES`:  strconv.Itoa(self.DeviceID),
			`PYTHONUNBUFFERED`:      `1`,
			// TODO: add LD_PRELOAD to tcmalloc path
			// `LD_PRELOAD`:``,
		}
		if disableNCCL {
			envs[`KUNGFU_DISABLE_NCCL`] = `1`
		}
		ps = append(ps, Proc{
			Name:    name,
			Prog:    prog,
			Args:    args,
			Envs:    merge(configEnvs, envs),
			Host:    self.NetAddr.Host,
			PubAddr: self.NetAddr.Host,
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
