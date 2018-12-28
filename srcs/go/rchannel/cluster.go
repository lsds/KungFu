package rchannel

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"

	"github.com/luomai/kungfu/srcs/go/log"
)

const ClusterSpecEnvKey = `KF_CLUSTER_SPEC`

// FIXME: make members private, public is required by JSON encoding for now

type TaskSpec struct {
	DeviceID       int // local rank
	NetAddr        NetAddr
	MonitoringPort uint16

	GlobalRank int // FIXME: make it dynamic
	HostID     int

	SockFile string
}

type ClusterSpec struct {
	Self  TaskSpec
	Peers []TaskSpec

	Root        int
	GatherGraph *Graph
	BcastGraph  *Graph
}

func (c ClusterSpec) String() string {
	bs, err := json.Marshal(c)
	if err != nil {
		return ""
	}
	return string(bs)
}

type Cluster struct {
	spec ClusterSpec
}

func NewClusterFromEnv() (*Cluster, error) {
	config := os.Getenv(ClusterSpecEnvKey)
	if len(config) == 0 {
		specs := GenClusterSpecs(1, []string{`127.0.0.1`}, 1)
		return &Cluster{spec: specs[0]}, nil
	}
	var spec ClusterSpec
	if err := json.Unmarshal([]byte(config), &spec); err != nil {
		return nil, errors.New(ClusterSpecEnvKey + " is invalid")
	}
	return &Cluster{spec: spec}, nil
}

func (c Cluster) Size() int {
	return len(c.spec.Peers)
}

func (c Cluster) Root() int {
	return c.spec.Root
}

func (c Cluster) GetPeer(rank int) TaskSpec {
	return c.spec.Peers[rank]
}

func (c Cluster) AllPeers() []TaskSpec {
	return c.spec.Peers
}

func (c Cluster) MyPort() uint32 {
	port, err := strconv.Atoi(c.spec.Self.NetAddr.Port)
	if err != nil {
		return 0
	}
	return uint32(port)
}

func (c Cluster) MyMonitoringPort() uint16 {
	return c.spec.Self.MonitoringPort
}

func (c Cluster) MyRank() int {
	return c.spec.Self.GlobalRank
}

func (c Cluster) PrevAndNext(i int) (int, int) {
	// TODO: deprecate
	// only for circular reduce
	k := c.Size()
	prev := (i - 1 + k) % k
	next := (i + 1 + k) % k
	return prev, next
}

func (c Cluster) Neighbours(i int) ([]int, []int, []int, []int) {
	n1 := c.spec.GatherGraph.Nodes[i]
	n2 := c.spec.BcastGraph.Nodes[i]
	return n1.Prevs, n1.Nexts, n2.Prevs, n2.Nexts
}

func GenClusterSpecs(k int, hosts []string, m int) []ClusterSpec {
	if cap := m * len(hosts); cap < k {
		log.Warnf("can run %d tasks at most!", cap)
	}
	tasks, gIn, gOut := genTaskSpecs(k, hosts, m)
	var specs []ClusterSpec
	for _, task := range tasks {
		spec := ClusterSpec{
			Self:        task,
			Peers:       tasks,
			Root:        0,
			GatherGraph: gIn,
			BcastGraph:  gOut,
		}
		specs = append(specs, spec)
	}
	return specs
}

func genTaskSpecs(k int, hosts []string, m int) ([]TaskSpec, *Graph, *Graph) {
	rankOf := func(hostID, deviceID int) int {
		return hostID*m + deviceID
	}

	g1 := newGraph(k)
	g2 := newGraph(k)

	addEdgesTo := func(t TaskSpec, fatherID int) {
		g1.AddEdge(t.GlobalRank, fatherID)
		g2.AddEdge(fatherID, t.GlobalRank)
	}

	addEdges := func(t TaskSpec) {
		if t.HostID == 0 && t.DeviceID == 0 {
			return
		}
		if t.DeviceID == 0 {
			addEdgesTo(t, rankOf(0, 0))
			return
		}
		addEdgesTo(t, rankOf(t.HostID, 0))
	}

	var tasks []TaskSpec
	for i, host := range hosts {
		for j := 0; j < m; j++ {
			port := strconv.Itoa(10001 + j)
			task := TaskSpec{
				HostID:   i,
				DeviceID: j,
				NetAddr: NetAddr{
					Host: host,
					Port: port,
				},
				MonitoringPort: uint16(20001 + j),
				SockFile:       sockFileFor(port),
				GlobalRank:     rankOf(i, j),
			}
			tasks = append(tasks, task)
			addEdges(task)
			if len(tasks) >= k {
				return tasks, g1, g2
			}
		}
	}
	log.Infof("only generated %d at best effort, instead of %d", len(tasks), k)
	return tasks, g1, g2
}

func sockFileFor(port string) string {
	return fmt.Sprintf(`/tmp/kungfu-run-%s.sock`, port)
}
