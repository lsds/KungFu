package plan

import (
	"fmt"
	"net"
	"os"
	"strconv"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

// PeerID is the unique identifier of a peer.
type PeerID NetAddr

func (p PeerID) String() string {
	return NetAddr(p).String()
}

func (p PeerID) ColocatedWith(q PeerID) bool {
	return NetAddr(p).ColocatedWith(NetAddr(q))
}

func (p PeerID) WithName(name string) Addr {
	return NetAddr(p).WithName(name)
}

func (p PeerID) SockFile() string {
	return NetAddr(p).SockFile()
}

func parseID(val string) (*PeerID, error) {
	host, p, err := net.SplitHostPort(val)
	if err != nil {
		return nil, err
	}
	ipv4, err := ParseIPv4(host) // FIXME: checkout error
	if err != nil {
		return nil, err
	}
	port, err := strconv.Atoi(p)
	if err != nil {
		return nil, err
	}
	if int(uint16(port)) != port {
		return nil, errInvalidPort
	}
	return &PeerID{
		Host: FormatIPv4(ipv4),
		Port: uint16(port),
	}, nil
}

func GetSelfFromEnv() (*PeerID, error) {
	config, ok := os.LookupEnv(kb.SelfSpecEnvKey)
	if !ok {
		ps := genPeerIDs(1, []HostSpec{DefaultHostSpec()})
		return &ps[0], nil
	}
	return parseID(config)
}

func GetParentFromEnv() (*PeerID, error) {
	val, ok := os.LookupEnv(kb.ParentIDEnvKey)
	if !ok {
		return nil, fmt.Errorf("%s not set", kb.ParentIDEnvKey)
	}
	return parseID(val)
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
				Port: uint16(10000 + j),
			}
			peers = append(peers, peer)
			if len(peers) >= k {
				return peers
			}
		}
	}
	return peers
}
