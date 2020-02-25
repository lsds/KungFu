package plan

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"strings"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

type PeerList []PeerID

func (pl PeerList) String() string {
	var parts []string
	for _, p := range pl {
		parts = append(parts, p.String())
	}
	return strings.Join(parts, ",")
}

func (pl PeerList) Bytes() []byte {
	b := &bytes.Buffer{}
	for _, p := range pl {
		binary.Write(b, binary.LittleEndian, &p)
	}
	return b.Bytes()
}

func (pl PeerList) Rank(ps PeerID) (int, bool) {
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

func (pl PeerList) Set() map[PeerID]struct{} {
	s := make(map[PeerID]struct{})
	for _, p := range pl {
		s[p] = struct{}{}
	}
	return s
}

func (pl PeerList) sub(ql PeerList) PeerList {
	s := ql.Set()
	var a PeerList
	for _, p := range pl {
		if _, ok := s[p]; !ok {
			a = append(a, p)
		}
	}
	return a
}

func (pl PeerList) Intersection(ql PeerList) PeerList {
	s := ql.Set()
	var a PeerList
	for _, p := range pl {
		if _, ok := s[p]; ok {
			a = append(a, p)
		}
	}
	return a
}

func (pl PeerList) Disjoint(ql PeerList) bool {
	return len(pl.Intersection(ql)) == 0
}

func (pl PeerList) Diff(ql PeerList) (PeerList, PeerList) {
	return pl.sub(ql), ql.sub(pl)
}

func (pl PeerList) Eq(ql PeerList) bool {
	if len(pl) != len(ql) {
		return false
	}
	for i, p := range pl {
		if p != ql[i] {
			return false
		}
	}
	return true
}

func (pl PeerList) On(host uint32) PeerList {
	var ql PeerList
	for _, p := range pl {
		if p.IPv4 == host {
			ql = append(ql, p)
		}
	}
	return ql
}

func parsePeerList(val string) (PeerList, error) {
	parts := strings.Split(val, ",")
	var pl PeerList
	for _, p := range parts {
		id, err := parseID(p)
		if err != nil {
			return nil, err
		}
		pl = append(pl, *id)
	}
	return pl, nil
}

func getInitPeersFromEnv() (PeerList, error) {
	val, ok := os.LookupEnv(kb.PeerListEnvKey)
	if !ok {
		return nil, fmt.Errorf("%s not set", kb.PeerListEnvKey)
	}
	return parsePeerList(val)
}
