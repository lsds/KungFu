package plan

import (
	"os"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

type Env struct {
	Self      PeerID
	Parent    PeerID
	InitPeers PeerList
	Strategy  kb.Strategy

	// resources
	HostList  HostList
	PortRange PortRange
}

func ParseEnv() (*Env, error) {
	if _, ok := os.LookupEnv(kb.SelfSpecEnvKey); !ok {
		return singleEnv(), nil
	}
	self, err := getSelfFromEnv()
	if err != nil {
		return nil, err
	}
	parent, err := getParentFromEnv()
	if err != nil {
		return nil, err
	}
	hostList, err := getHostListFromEnv()
	if err != nil {
		return nil, err
	}
	portRange, err := getPortRangeFromEnv()
	if err != nil {
		return nil, err
	}
	initPeers, err := getInitPeersFromEnv()
	if err != nil {
		return nil, err
	}
	strategy, err := kb.ParseStrategy(os.Getenv(kb.AllReduceStrategyEnvKey))
	if err != nil {
		return nil, err
	}
	return &Env{
		Self:      *self,
		Parent:    *parent,
		HostList:  hostList,
		PortRange: *portRange,
		InitPeers: initPeers,
		Strategy:  *strategy,
	}, nil
}

func singleEnv() *Env {
	self := DefaultHostList.genPeerList(1, DefaultPortRange)[0]
	return &Env{
		Self:      self,
		InitPeers: PeerList{self},
		Strategy:  kb.DefaultStrategy,
	}
}
