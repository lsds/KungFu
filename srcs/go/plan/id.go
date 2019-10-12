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
		IPv4: ipv4,
		Port: uint16(port),
	}, nil
}

func getSelfFromEnv() (*PeerID, error) {
	config, ok := os.LookupEnv(kb.SelfSpecEnvKey)
	if !ok {
		return nil, fmt.Errorf("%s not set", kb.SelfSpecEnvKey)
	}
	return parseID(config)
}

func getParentFromEnv() (*PeerID, error) {
	val, ok := os.LookupEnv(kb.ParentIDEnvKey)
	if !ok {
		return nil, fmt.Errorf("%s not set", kb.ParentIDEnvKey)
	}
	return parseID(val)
}
