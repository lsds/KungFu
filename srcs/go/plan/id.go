package plan

import (
	"net"
	"strconv"
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

func (p PeerID) ListenAddr(strict bool) NetAddr {
	if strict {
		return NetAddr{IPv4: p.IPv4, Port: p.Port}
	}
	return NetAddr{IPv4: 0, Port: p.Port}
}

func (p PeerID) SockFile() string {
	return NetAddr(p).SockFile()
}

func ParsePeerID(val string) (*PeerID, error) {
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
		IPv4: ipv4,
		Port: uint16(port),
	}, nil
}
