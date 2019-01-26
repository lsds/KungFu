package plan

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

// FIXME: make members private, public is required by JSON encoding for now

type ProcSpec struct {
	ClusterSpec
	SelfRank int
}

func (ps ProcSpec) Self() PeerSpec {
	return ps.Peers[ps.SelfRank]
}

func NewProcSpecFromEnv() (*ProcSpec, error) {
	clusterSpecConfig := os.Getenv(kb.ClusterSpecEnvKey)
	selfRankConfig := os.Getenv(kb.SelfRankEnvKey)
	if len(clusterSpecConfig) == 0 && len(selfRankConfig) == 0 {
		cs, err := GenClusterSpec(1, []HostSpec{DefaultHostSpec()})
		if err != nil {
			return nil, err
		}
		ps := cs.ToProcSpec(0)
		return &ps, nil
	}

	var cs ClusterSpec
	if err := json.Unmarshal([]byte(clusterSpecConfig), &cs); err != nil {
		return nil, errors.New(kb.ClusterSpecEnvKey + " is invalid")
	}
	selfRank, err := strconv.Atoi(selfRankConfig)
	if err != nil {
		return nil, errors.New(kb.SelfRankEnvKey + " is invalid")
	}
	if selfRank < 0 || len(cs.Peers) <= selfRank {
		return nil, errors.New(kb.SelfRankEnvKey + " is invalid")
	}
	ps := cs.ToProcSpec(selfRank)
	return &ps, nil
}

func (ps ProcSpec) Size() int {
	return len(ps.Peers)
}

func (ps ProcSpec) GetPeer(rank int) PeerSpec {
	return ps.Peers[rank]
}

func (ps ProcSpec) AllPeers() []PeerSpec {
	return ps.Peers
}

func (ps ProcSpec) MyRank() int {
	return ps.SelfRank
}

func GenClusterSpec(k int, hostSpecs []HostSpec) (*ClusterSpec, error) {
	if cap := TotalCap(hostSpecs); cap < k {
		return nil, fmt.Errorf("can run %d peers at most!", cap)
	}
	return &ClusterSpec{Peers: genPeerSpecs(k, hostSpecs)}, nil
}

func genPeerSpecs(k int, hostSpecs []HostSpec) []PeerSpec {
	var peers []PeerSpec
	for _, host := range hostSpecs {
		for j := 0; j < host.Slots; j++ {
			peer := PeerSpec{
				DeviceID: j,
				NetAddr: NetAddr{
					Host: host.Hostname,
					Port: uint16(10001 + j),
				},
				MonitoringPort: uint16(20001 + j),
			}
			peers = append(peers, peer)
			if len(peers) >= k {
				return peers
			}
		}
	}
	return peers
}
