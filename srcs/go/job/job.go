package job

import (
	"fmt"
	"os"
	"strconv"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/kungfu/env"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type Job struct {
	ConfigServer string
	Strategy     kb.Strategy
	Parent       plan.PeerID
	HostList     plan.HostList
	PortRange    plan.PortRange
	Prog         string
	Args         []string
	LogDir       string

	AllowNVLink bool
}

func (j Job) NewProc(peer plan.PeerID, gpuID int, initClusterVersion int, pl plan.PeerList) Proc {
	envs := Envs{
		env.SelfSpecEnvKey:           peer.String(),
		env.HostListEnvKey:           j.HostList.String(),
		env.ParentIDEnvKey:           j.Parent.String(),
		env.PeerListEnvKey:           pl.String(),
		env.InitClusterVersionEnvKey: strconv.Itoa(initClusterVersion),
		env.AllReduceStrategyEnvKey:  j.Strategy.String(),
		env.ConfigServerEnvKey:       j.ConfigServer,
	}
	if len(j.ConfigServer) > 0 {
		envs[env.ConfigServerEnvKey] = j.ConfigServer
	}
	cudaIdx := strconv.Itoa(getCudaIndex(gpuID))
	envs[`KUNGFU_`+cudaVisibleDevicesKey] = cudaIdx
	if j.AllowNVLink {
		log.Warnf("Please set `config.gpu_options.visible_device_list = str(local_rank)`")
	} else {
		envs[cudaVisibleDevicesKey] = cudaIdx
	}

	allEnvs := merge(getConfigEnvs(), envs)
	allEnvs.addIfMissing(`PYTHONUNBUFFERED`, `1`)
	var pubAddr string
	for _, h := range j.HostList {
		if h.IPv4 == peer.IPv4 {
			pubAddr = h.PublicAddr
		}
	}

	return Proc{
		Name:     fmt.Sprintf("%s.%d", plan.FormatIPv4(peer.IPv4), peer.Port),
		Prog:     j.Prog,
		Args:     j.Args,
		Envs:     allEnvs,
		Hostname: pubAddr,
		LogDir:   j.LogDir,
	}
}

func (j Job) CreateAllProcs(pl plan.PeerList) []Proc {
	var ps []Proc
	for _, self := range pl {
		localRank, _ := pl.LocalRank(self)
		proc := j.NewProc(self, localRank, 0, pl)
		ps = append(ps, proc)
	}
	return ps
}

func (j Job) CreateProcs(pl plan.PeerList, host uint32) []Proc {
	var ps []Proc
	for _, self := range pl.On(host) {
		localRank, _ := pl.LocalRank(self)
		proc := j.NewProc(self, localRank, 0, pl)
		ps = append(ps, proc)
	}
	return ps
}

func getConfigEnvs() Envs {
	envs := make(Envs)
	for _, k := range config.ConfigEnvKeys {
		if val := os.Getenv(k); len(val) > 0 {
			envs[k] = val
		}
	}
	return envs
}
