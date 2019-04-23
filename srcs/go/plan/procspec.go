package plan

import (
	"errors"
	"fmt"
	"os"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

// FIXME: make members private, public is required by JSON encoding for now

type ProcSpec struct {
	ClusterSpec
	self PeerSpec
}

func (ps ProcSpec) Self() PeerSpec {
	return ps.self
}

func defaultProcSpec() (*ProcSpec, error) {
	cs, err := GenClusterSpec(1, []HostSpec{DefaultHostSpec()})
	if err != nil {
		return nil, err
	}
	return &ProcSpec{
		ClusterSpec: *cs,
		self:        cs.Peers[0],
	}, nil
}

func GetSelfFromEnv() (*PeerSpec, error) {
	self, err := getSelfFromEnv()
	if err != nil {
		ps, err := defaultProcSpec()
		if err != nil {
			return nil, err
		}
		return &ps.self, nil
	}
	return self, nil
}

func NewProcSpecFromEnv() (*ProcSpec, error) {
	self, err := getSelfFromEnv()
	if err != nil {
		return defaultProcSpec()
	}
	clusterSpecConfig := os.Getenv(kb.ClusterSpecEnvKey)
	var cs ClusterSpec
	if err := fromString(clusterSpecConfig, &cs); err != nil {
		return nil, err
	}
	ps := ProcSpec{
		ClusterSpec: cs,
		self:        *self,
	}
	if _, err := ps.MyRank(); err != nil {
		return nil, err
	}
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

func (ps ProcSpec) MyRank() (int, error) {
	for i, p := range ps.Peers {
		if p == ps.self {
			return i, nil
		}
	}
	return -1, errors.New("self not in cluster")
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
