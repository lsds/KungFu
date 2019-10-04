package plan

import (
	"os"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

// PeerSpec describes the system resources that will be used by the process of the peer
type PeerSpec struct {
	NetAddr NetAddr
}

func (ps PeerSpec) ColocatedWith(q PeerSpec) bool {
	return ps.NetAddr.Host == q.NetAddr.Host
}

func (ps PeerSpec) String() string {
	return toString(ps)
}

func GetSelfFromEnv() (*PeerSpec, error) {
	config := os.Getenv(kb.SelfSpecEnvKey)
	if len(config) == 0 {
		ps := genPeerSpecs(1, []HostSpec{DefaultHostSpec()})
		return &ps[0], nil
	}
	var ps PeerSpec
	if err := FromString(config, &ps); err != nil {
		return nil, err
	}
	return &ps, nil
}

func genPeerSpecs(k int, hostSpecs []HostSpec) []PeerSpec {
	if k == 0 {
		return nil
	}
	var peers []PeerSpec
	for _, host := range hostSpecs {
		for j := 0; j < host.Slots; j++ {
			peer := PeerSpec{
				NetAddr: NetAddr{
					Host: host.Hostname,
					Port: uint16(10001 + j),
				},
			}
			peers = append(peers, peer)
			if len(peers) >= k {
				return peers
			}
		}
	}
	return peers
}
