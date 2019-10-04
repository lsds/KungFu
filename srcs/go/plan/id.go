package plan

import (
	"os"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

// PeerID is the unique identifier of a peer.
type PeerID NetAddr

func (p PeerID) String() string {
	return toString(p)
}

func (p PeerID) ColocatedWith(q PeerID) bool {
	return NetAddr(p).ColocatedWith(NetAddr(q))
}

func (p PeerID) WithName(name string) Addr {
	return NetAddr(p).WithName(name)
}

func GetSelfFromEnv() (*PeerID, error) {
	config := os.Getenv(kb.SelfSpecEnvKey)
	if len(config) == 0 {
		ps := genPeerIDs(1, []HostSpec{DefaultHostSpec()})
		return &ps[0], nil
	}
	var ps PeerID
	if err := FromString(config, &ps); err != nil {
		return nil, err
	}
	return &ps, nil
}

func genPeerIDs(k int, hostSpecs []HostSpec) []PeerID {
	if k == 0 {
		return nil
	}
	var peers []PeerID
	for _, host := range hostSpecs {
		for j := 0; j < host.Slots; j++ {
			peer := PeerID{
				Host: host.Hostname,
				Port: uint16(10001 + j),
			}
			peers = append(peers, peer)
			if len(peers) >= k {
				return peers
			}
		}
	}
	return peers
}
