package huawei

import (
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"strconv"

	"github.com/lsds/KungFu/srcs/go/plan"
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
	SelfIPv4       string
	ClusterSpec    *plan.ClusterSpec
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
	clusterSpec, err := parseClusterSpec(num)
	if err != nil {
		return nil, err
	}
	return &ContainerInfo{
		ContainerIndex: idx,
		ClusterSize:    num,
		SelfIPv4:       clusterSpec.Peers[idx].NetAddr.Host,
		ClusterSpec:    clusterSpec,
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

func parseClusterSpec(n int) (*plan.ClusterSpec, error) {
	var peers []plan.PeerSpec
	for i := 0; i < n; i++ {
		key := fmt.Sprintf(PeerAddrFormat, i)
		val := os.Getenv(key)
		if len(val) <= 0 {
			return nil, fmt.Errorf("%s not set", key)
		}
		ipv4, port, err := resolvePeer(val)
		if err != nil {
			return nil, err
		}
		peer := plan.PeerSpec{
			DeviceID: 0,
			NetAddr: plan.NetAddr{
				Host: ipv4,
				Port: uint16(port),
			},
			MonitoringPort: uint16(20001),
		}
		log.Printf("peer: %d: %#v", i, peer)
		peers = append(peers, peer)
	}
	return &plan.ClusterSpec{Peers: peers}, nil
}

func resolvePeer(hostPort string) (string, int, error) {
	h, p, err := net.SplitHostPort(hostPort)
	if err != nil {
		return "", 0, err
	}
	addrs, err := net.LookupHost(h)
	if err != nil {
		return "", 0, err
	}
	if len(addrs) != 1 {
		return "", 0, errors.New("exactly 1 addr is expected")
	}
	port, err := strconv.Atoi(p)
	if len(addrs) != 1 {
		return "", 0, err
	}
	return addrs[0], port, nil
}
