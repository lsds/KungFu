package env

import (
	"encoding/json"
	"net"
	"strconv"

	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type peerID plan.PeerID // customized JSON encoding

func (p peerID) MarshalJSON() ([]byte, error) {
	port := strconv.Itoa(int(p.Port))
	addr := net.JoinHostPort(plan.FormatIPv4(p.IPv4), port)
	return json.Marshal(addr)
}

func (p *peerID) UnmarshalJSON(bs []byte) error {
	var s string
	if err := json.Unmarshal(bs, &s); err != nil {
		return err
	}
	id, err := plan.ParsePeerID(s)
	if err != nil {
		return err
	}
	*p = peerID(*id)
	return nil
}

type clusterSpec struct {
	Peers []peerID `json:"peers"`
}

type peerSpec struct {
	Rank int `json:"rank"`
}

type kungfuConfig struct {
	Cluster clusterSpec `json:"cluster"`
	Self    peerSpec    `json:"self"`
}

func ParseConfigFromJSON(js string) (*Config, error) {
	var kfConfig kungfuConfig
	if err := json.Unmarshal([]byte(js), &kfConfig); err != nil {
		return nil, err
	}
	var initPeers plan.PeerList
	for _, p := range kfConfig.Cluster.Peers {
		initPeers = append(initPeers, plan.PeerID(p))
	}
	return &Config{
		InitPeers: initPeers,
		Self:      initPeers[kfConfig.Self.Rank],
		Strategy:  base.DefaultStrategy,
	}, nil
}
