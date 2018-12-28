package rchannel

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
)

const ClusterSpecEnvKey = `KF_CLUSTER_SPEC`

// FIXME: make members private, public is required by JSON encoding for now

var errInvalidHostSpec = errors.New("Invalid HostSpec")

type HostSpec struct {
	Hostname   string
	Slots      int
	PublicAddr string
}

func DefaultHostSpec() HostSpec {
	return HostSpec{
		Hostname:   `127.0.0.1`,
		Slots:      runtime.NumCPU(),
		PublicAddr: `127.0.0.1`,
	}
}

func (h HostSpec) String() string {
	return fmt.Sprintf("%s:%d:%s", h.Hostname, h.Slots, h.PublicAddr)
}

func parseHostSpec(spec string) (*HostSpec, error) {
	parts := strings.Split(spec, ":")
	switch len(parts) {
	case 1:
		return &HostSpec{Hostname: parts[0], Slots: 1, PublicAddr: parts[0]}, nil
	case 2:
		slots, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, errInvalidHostSpec
		}
		return &HostSpec{Hostname: parts[0], Slots: slots, PublicAddr: parts[0]}, nil
	case 3:
		slots, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, errInvalidHostSpec
		}
		return &HostSpec{Hostname: parts[0], Slots: slots, PublicAddr: parts[2]}, nil
	}
	return nil, errInvalidHostSpec
}

func ParseHostSpec(h string) ([]HostSpec, error) {
	var hostSpecs []HostSpec
	for _, h := range strings.Split(h, ",") {
		spec, err := parseHostSpec(h)
		if err != nil {
			return nil, err
		}
		hostSpecs = append(hostSpecs, *spec)
	}
	return hostSpecs, nil
}

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

func getConfig() string {
	config := os.Getenv(ClusterSpecEnvKey)
	if len(config) != 0 {
		return config
	}
	if specs, err := GenClusterSpecs(1, []HostSpec{DefaultHostSpec()}); err == nil {
		return specs[0].String()
	}
	return ""
}

func NewClusterFromEnv() (*Cluster, error) {
	config := getConfig()
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

func totalCap(hostSpecs []HostSpec) int {
	var cap int
	for _, h := range hostSpecs {
		cap += h.Slots
	}
	return cap
}

func GenClusterSpecs(k int, hostSpecs []HostSpec) ([]ClusterSpec, error) {
	if cap := totalCap(hostSpecs); cap < k {
		return nil, fmt.Errorf("can run %d tasks at most!", cap)
	}
	tasks, gIn, gOut := genTaskSpecs(k, hostSpecs)
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
	return specs, nil
}

func genTaskSpecs(k int, hostSpecs []HostSpec) ([]TaskSpec, *Graph, *Graph) {
	var tasksBefore []int
	{
		before := 0
		for _, h := range hostSpecs {
			tasksBefore = append(tasksBefore, before)
			before += h.Slots
		}
	}

	rankOf := func(hostID, deviceID int) int {
		return tasksBefore[hostID] + deviceID
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
	for i, host := range hostSpecs {
		for j := 0; j < host.Slots; j++ {
			port := strconv.Itoa(10001 + j)
			task := TaskSpec{
				HostID:   i,
				DeviceID: j,
				NetAddr: NetAddr{
					Host: host.Hostname,
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
	return tasks, g1, g2
}

func sockFileFor(port string) string {
	return fmt.Sprintf(`/tmp/kungfu-run-%s.sock`, port)
}
