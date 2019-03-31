package huawei

import (
	"fmt"
	"os"
	"strconv"
)

// https://github.com/huawei-clouds/modelarts-example/blob/master/CustomImage/自定义镜像训练功能操作指南.md
const (
	TaskIndexEnvKey = `DLS_TASK_INDEX`
	TaskNumEnvKey   = `DLS_TASK_NUMBER`
	PeerAddrFormat  = `BATCH_CUSTOM%d_HOSTS`
)

type ContainerInfo struct {
	ContainerIndex int
	ClusterSize    int
	Peers          []string
}

func ParseEnv() (*ContainerInfo, error) {
	idx, err := requireInt(TaskIndexEnvKey)
	if err != nil {
		return nil, err
	}
	num, err := requireInt(TaskNumEnvKey)
	if err != nil {
		return nil, err
	}
	peers, err := parsePeers(num)
	return &ContainerInfo{
		ContainerIndex: idx,
		ClusterSize:    num,
		Peers:          peers,
	}, nil
}

func requireInt(key string) (int, error) {
	val := os.Getenv(key)
	if len(val) <= 0 {
		return 0, fmt.Errorf("%s not set", key)
	}
	n, err := strconv.Atoi(val)
	if err != nil {
		return 0, err
	}
	return n, nil
}

func parsePeers(n int) ([]string, error) {
	var ips []string
	for i := 0; i < n; i++ {
		key := fmt.Sprintf(PeerAddrFormat, i)
		val := os.Getenv(key)
		if len(val) <= 0 {
			return nil, fmt.Errorf("%s not set", key)
		}
		ips = append(ips, val)
	}
	return ips, nil
}
