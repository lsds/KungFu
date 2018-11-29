package rchannel

import (
	"encoding/json"
	"errors"
	"os"
	"strconv"
)

const ClusterSpecEnvKey = `KF_CLUSTER_SPEC`

type ClusterSpec struct {
	Self  NetAddr
	Peers []NetAddr
}

func NewClusterSpecFromEnv() (*ClusterSpec, error) {
	config := os.Getenv(ClusterSpecEnvKey)
	if len(config) == 0 {
		clusters := GenCluster(1)
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
	port, err := strconv.Atoi(c.Self.Port)
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

func GenCluster(n int) []ClusterSpec {
	var peers []NetAddr
	for i := 0; i < n; i++ {
		peers = append(peers, NetAddr{
			Host: `127.0.0.1`,
			Port: strconv.Itoa(10001 + i),
		})
	}
	var specs []ClusterSpec
	for i := 0; i < n; i++ {
		spec := ClusterSpec{
			Self:  peers[i],
			Peers: peers,
		}
		specs = append(specs, spec)
	}
	return specs
}
