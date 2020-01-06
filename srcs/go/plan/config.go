package plan

import (
	"os"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

type Config struct {
	Parent   PeerID
	Parents  PeerList
	Self     PeerID
	Strategy kb.Strategy

	InitCheckpoint string
	InitPeers      PeerList

	// resources
	HostList  HostList
	PortRange PortRange

	Single bool
}

func ParseConfigFromEnv() (*Config, error) {
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
	return &Config{
		Self:           *self,
		Parent:         *parent,
		Parents:        getParentIDs(hostList, *parent),
		HostList:       hostList,
		PortRange:      *portRange,
		InitPeers:      initPeers,
		Strategy:       *strategy,
		InitCheckpoint: os.Getenv(kb.InitStepEnvKey),
	}, nil
}

func getParentIDs(hl HostList, parent PeerID) PeerList {
	var ps PeerList
	for _, h := range hl {
		ps = append(ps, PeerID{IPv4: h.IPv4, Port: parent.Port})
	}
	return ps
}

func singleEnv() *Config {
	self := DefaultHostList.genPeerList(1, DefaultPortRange)[0]
	return &Config{
		Self:      self,
		InitPeers: PeerList{self},
		Strategy:  kb.DefaultStrategy,
		Single:    true,
	}
}
