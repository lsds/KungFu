package huawei

import (
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func ParseEnvV100(np int) (*ContainerInfo, error) {
	gpus := utils.ListNvidiaGPUNames()
	clusterSpec, err := genV100ClusterSpec(np)
	if err != nil {
		return nil, err
	}
	return &ContainerInfo{
		SelfIPv4:    `127.0.0.1`,
		GPUs:        gpus,
		ClusterSpec: clusterSpec,
	}, nil
}

func genHostSpec(ibIPv4 string, gpus int) plan.HostSpec {
	return plan.HostSpec{
		Hostname:   ibIPv4,
		Slots:      gpus,
		PublicAddr: ibIPv4,
	}
}

func genV100ClusterSpec(np int) (*plan.ClusterSpec, error) {
	return plan.GenClusterSpec(np, []plan.HostSpec{
		genHostSpec(`127.0.0.1`, 8),
	})
}
