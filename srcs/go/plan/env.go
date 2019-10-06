package plan

import (
	"os"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

type Env struct {
	Self      PeerID
	Parent    PeerID
	HostList  HostList
	InitPeers PeerList
}

func ParseEnv() (*Env, error) {
	if _, ok := os.LookupEnv(kb.SelfSpecEnvKey); !ok {
		return singleEnv(), nil
	}
	self, err := GetSelfFromEnv()
	if err != nil {
		return nil, err
	}
	parent, err := GetParentFromEnv()
	if err != nil {
		return nil, err
	}
	hostList, err := GetHostListFromEnv()
	if err != nil {
		return nil, err
	}
	InitPeers, err := GetInitPeersFromEnv()
	if err != nil {
		return nil, err
	}
	return &Env{
		Self:      *self,
		Parent:    *parent,
		HostList:  hostList,
		InitPeers: InitPeers,
	}, nil
}

func singleEnv() *Env {
	hl := HostList{DefaultHostSpec()}
	self := hl.genPeerList(1, DefaultPortRange)[0]
	return &Env{
		Self:      self,
		InitPeers: PeerList{self},
	}
}
