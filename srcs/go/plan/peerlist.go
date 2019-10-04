package plan

import "fmt"

type PeerList []PeerID

func (pl PeerList) String() string {
	return toString(pl)
}

func (pl PeerList) Lookup(ps PeerID) (int, bool) {
	for i, p := range pl {
		if p == ps {
			return i, true
		}
	}
	return -1, false
}

func (pl PeerList) LocalRank(ps PeerID) (int, bool) {
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

func GenPeerList(k int, hostSpecs []HostSpec) (PeerList, error) {
	if cap := TotalCap(hostSpecs); cap < k {
		return nil, fmt.Errorf("can run %d peers at most", cap)
	}
	return genPeerIDs(k, hostSpecs), nil
}
