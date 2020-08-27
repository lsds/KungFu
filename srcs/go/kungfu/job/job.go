package job

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/kungfu/env"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/proc"
)

type Job struct {
	StartTime    time.Time
	ConfigServer string
	Strategy     base.Strategy
	Parent       plan.PeerID
	HostList     plan.HostList
	PortRange    plan.PortRange
	Prog         string
	Args         []string
	LogDir       string

	AllowNVLink bool
}

func (j Job) NewProc(peer plan.PeerID, gpuID int, initClusterVersion int, cluster plan.Cluster) proc.Proc {
	envs := proc.Envs{
		env.JobStartTimestamp:        strconv.FormatInt(j.StartTime.Unix(), 10),
		env.ProcStartTimestamp:       strconv.FormatInt(time.Now().Unix(), 10),
		env.SelfSpecEnvKey:           peer.String(),
		env.RunnerListEnvKey:         cluster.Runners.String(),
		env.ParentIDEnvKey:           j.Parent.String(),
		env.PeerListEnvKey:           cluster.Workers.String(),
		env.InitClusterVersionEnvKey: strconv.Itoa(initClusterVersion),
		env.AllReduceStrategyEnvKey:  j.Strategy.String(),
		env.ConfigServerEnvKey:       j.ConfigServer,
		env.AllowNvLink:              fmt.Sprintf("%v", j.AllowNVLink),
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

	allEnvs := proc.Merge(getConfigEnvs(), envs)
	allEnvs.AddIfMissing(`PYTHONUNBUFFERED`, `1`)
	var pubAddr string
	for _, h := range j.HostList {
		if h.IPv4 == peer.IPv4 {
			pubAddr = h.PublicAddr
		}
	}

	return proc.Proc{
		Name:     fmt.Sprintf("%s.%d", plan.FormatIPv4(peer.IPv4), peer.Port),
		Prog:     j.Prog,
		Args:     j.Args,
		Envs:     allEnvs,
		Hostname: pubAddr,
		LogDir:   j.LogDir,
	}
}

func (j Job) CreateProcs(cluster plan.Cluster, host uint32) []proc.Proc {
	var ps []proc.Proc
	for _, self := range cluster.Workers.On(host) {
		localRank, _ := cluster.Workers.LocalRank(self)
		proc := j.NewProc(self, localRank, 0, cluster)
		ps = append(ps, proc)
	}
	return ps
}

func (j Job) ProgAndArgs() []string {
	a := []string{j.Prog}
	a = append(a, j.Args...)
	return a
}

func getConfigEnvs() proc.Envs {
	envs := make(proc.Envs)
	for _, k := range config.ConfigEnvKeys {
		if val := os.Getenv(k); len(val) > 0 {
			envs[k] = val
		}
	}
	return envs
}

func (j Job) DebugString() string {
	return fmt.Sprintf("job{prog=%s, args=%q}", j.Prog, j.Args)
}
