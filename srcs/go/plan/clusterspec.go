package plan

import (
	"fmt"
)

type PeerList []PeerSpec

type ClusterSpec struct {
	Peers []PeerSpec
}

func (cs ClusterSpec) String() string {
	return toString(cs)
}

func (cs ClusterSpec) Lookup(ps PeerSpec) (int, bool) {
	for i, p := range cs.Peers {
		if p == ps {
			return i, true
		}
	}
	return -1, false
}

func (pl PeerList) LocalRank(ps PeerSpec) (int, bool) {
	var i int
	for _, p := range pl {
		if p == ps {
			return i, true
		}
		if ps.ColocatedWith(p) {
			i++
		}
	}
	return -1, false
}

func LocalRank(pl PeerList, ps PeerSpec) (int, bool) {
	return pl.LocalRank(ps)
}

func GenClusterSpec(k int, hostSpecs []HostSpec) (*ClusterSpec, error) {
	if cap := TotalCap(hostSpecs); cap < k {
		return nil, fmt.Errorf("can run %d peers at most", cap)
	}
	return &ClusterSpec{Peers: genPeerSpecs(k, hostSpecs)}, nil
}
