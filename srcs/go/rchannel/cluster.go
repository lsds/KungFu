package rchannel

import (
	"encoding/json"
	"errors"
	"os"
	"strconv"

	"github.com/luomai/kungfu/srcs/go/log"
)

const ClusterSpecEnvKey = `KF_CLUSTER_SPEC`

type TaskSpec struct {
	DeviceID       int
	NetAddr        NetAddr
	MonitoringPort uint16
}

type ClusterSpec struct {
	Self  TaskSpec
	Peers []TaskSpec
}

func NewClusterSpecFromEnv() (*ClusterSpec, error) {
	config := os.Getenv(ClusterSpecEnvKey)
	if len(config) == 0 {
		clusters := GenCluster(1, []string{`127.0.0.1`}, 1)
		return &clusters[0], nil
	}
	var cluster ClusterSpec
	if err := json.Unmarshal([]byte(config), &cluster); err != nil {
		return nil, errors.New(ClusterSpecEnvKey + " Not set")
	}
	return &cluster, nil
}

func (c ClusterSpec) String() string {
	bs, err := json.Marshal(c)
	if err != nil {
		return ""
	}
	return string(bs)
}

func (c ClusterSpec) MyPort() uint32 {
	port, err := strconv.Atoi(c.Self.NetAddr.Port)
	if err != nil {
		return 0
	}
	return uint32(port)
}

func (c ClusterSpec) MyRank() int {
	for i, a := range c.Peers {
		if a == c.Self {
			return i
		}
	}
	panic("Self is not in the cluster")
}

func GenCluster(n int, hosts []string, m int) []ClusterSpec {
	if cap := m * len(hosts); cap < n {
		log.Warnf("can run %d tasks at most!", cap)
	}
	tasks := genCluster(n, hosts, m)
	var specs []ClusterSpec
	for _, task := range tasks {
		spec := ClusterSpec{
			Self:  task,
			Peers: tasks,
		}
		specs = append(specs, spec)
	}
	return specs
}

func genCluster(n int, hosts []string, m int) []TaskSpec {
	var tasks []TaskSpec
	for _, host := range hosts {
		for i := 0; i < m; i++ {
			t := TaskSpec{
				DeviceID: i,
				NetAddr: NetAddr{
					Host: host,
					Port: strconv.Itoa(10001 + i),
				},
				MonitoringPort: uint16(20001 + i),
			}
			tasks = append(tasks, t)
			if len(tasks) >= n {
				return tasks
			}
		}
	}
	return tasks
}
