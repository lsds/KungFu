package scheduler

import (
	"fmt"
	"strconv"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type JobConfig struct {
	TaskCount int
	HostList  string
	Prog      string
	Args      []string
}

func (jc JobConfig) CreateProcs(algo kb.KungFu_AllReduceAlgo) ([]Proc, error) {
	hostSpecs, err := plan.ParseHostSpec(jc.HostList)
	if err != nil {
		return nil, err
	}
	cs, err := plan.GenClusterSpec(jc.TaskCount, hostSpecs)
	if err != nil {
		return nil, err
	}
	pubAddr := make(map[string]string)
	for _, h := range hostSpecs {
		pubAddr[h.Hostname] = h.PublicAddr
	}
	var ps []Proc
	for i, self := range cs.Peers {
		name := fmt.Sprintf("%02s/%02d/%02d", self.NetAddr.Host, self.DeviceID, self.GlobalRank)
		ps = append(ps, Proc{
			Name: name,
			Prog: jc.Prog,
			Args: jc.Args,
			Envs: map[string]string{
				plan.ProcSpecEnvKey:         cs.ToProcSpec(i).String(),
				`CUDA_VISIBLE_DEVICES`:      strconv.Itoa(self.DeviceID),
				`PYTHONUNBUFFERED`:          `1`,
				kb.KungFu_AllReduceAlgo_Key: algo.String(),
			},
			Host:    self.NetAddr.Host,
			PubAddr: pubAddr[self.NetAddr.Host],
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
