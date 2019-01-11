package plan

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
)

const ProcSpecEnvKey = `KUNGFU_PROC_SPEC`

// FIXME: make members private, public is required by JSON encoding for now

type ProcSpec struct {
	ClusterSpec
	SelfRank int
}

func (ps ProcSpec) String() string {
	bs, err := json.Marshal(ps)
	if err != nil {
		return ""
	}
	return string(bs)
}

func (pc ProcSpec) Self() TaskSpec {
	return pc.Peers[pc.SelfRank]
}

func getConfig() string {
	config := os.Getenv(ProcSpecEnvKey)
	if len(config) != 0 {
		return config
	}
	if cs, err := GenClusterSpec(1, []HostSpec{DefaultHostSpec()}); err == nil {
		return cs.ToProcSpec(0).String()
	}
	return ""
}

func NewProcSpecFromEnv() (*ProcSpec, error) {
	config := getConfig()
	var pc ProcSpec
	if err := json.Unmarshal([]byte(config), &pc); err != nil {
		return nil, errors.New(ProcSpecEnvKey + " is invalid")
	}
	return &pc, nil
}

func (pc ProcSpec) Size() int {
	return len(pc.Peers)
}

func (pc ProcSpec) GetPeer(rank int) TaskSpec {
	return pc.Peers[rank]
}

func (pc ProcSpec) AllPeers() []TaskSpec {
	return pc.Peers
}

func (pc ProcSpec) MyPort() uint32 {
	port, err := strconv.Atoi(pc.Self().NetAddr.Port)
	if err != nil {
		return 0
	}
	return uint32(port)
}

func (pc ProcSpec) MyMonitoringPort() uint16 {
	return pc.Self().MonitoringPort
}

func (pc ProcSpec) MyRank() int {
	return pc.SelfRank
}

func GenClusterSpec(k int, hostSpecs []HostSpec) (*ClusterSpec, error) {
	if cap := TotalCap(hostSpecs); cap < k {
		return nil, fmt.Errorf("can run %d tasks at most!", cap)
	}
	tasks := genTaskSpecs(k, hostSpecs)
	return &ClusterSpec{Peers: tasks}, nil
}

func genTaskSpecs(k int, hostSpecs []HostSpec) []TaskSpec {
	var idx int
	var tasks []TaskSpec
	for _, host := range hostSpecs {
		for j := 0; j < host.Slots; j++ {
			port := strconv.Itoa(10001 + j)
			task := TaskSpec{
				DeviceID: j,
				NetAddr: NetAddr{
					Host: host.Hostname,
					Port: port,
				},
				MonitoringPort: uint16(20001 + j),
				SockFile:       SockFileFor(port),
				GlobalRank:     idx,
			}
			idx++
			tasks = append(tasks, task)
			if len(tasks) >= k {
				return tasks
			}
		}
	}
	return tasks
}

func SockFileFor(port string) string {
	return fmt.Sprintf(`/tmp/kungfu-prun-%s.sock`, port)
}
