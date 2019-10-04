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

func (pl PeerList) set() map[PeerID]struct{} {
	s := make(map[PeerID]struct{})
	for _, p := range pl {
		s[p] = struct{}{}
	}
	return s
}

func (pl PeerList) sub(ql PeerList) PeerList {
	s := ql.set()
	var a PeerList
	for _, p := range pl {
		if _, ok := s[p]; !ok {
			a = append(a, p)
		}
	}
	return a
}

func (pl PeerList) Diff(ql PeerList) (PeerList, PeerList) {
	return pl.sub(ql), ql.sub(pl)
}

func GenPeerList(k int, hostSpecs []HostSpec) (PeerList, error) {
	if cap := TotalCap(hostSpecs); cap < k {
		return nil, fmt.Errorf("can run %d peers at most", cap)
	}
	return genPeerIDs(k, hostSpecs), nil
}
