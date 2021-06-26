package env

import (
	"encoding/json"
	"net"
	"strconv"

	"github.com/lsds/KungFu/srcs/go/plan"
)

type taskID plan.PeerID

func (t taskID) MarshalJSON() ([]byte, error) {
	port := strconv.Itoa(int(t.Port))
	addr := net.JoinHostPort(plan.FormatIPv4(t.IPv4), port)
	return json.Marshal(addr)
}

func (t *taskID) UnmarshalJSON(bs []byte) error {
	var s string
	if err := json.Unmarshal(bs, &s); err != nil {
		return err
	}
	id, err := plan.ParsePeerID(s)
	if err != nil {
		return err
	}
	*t = taskID(*id)
	return nil
}

type clusterSpec struct {
	Worker []taskID `json:"worker"`
}

type taskSpec struct {
	Index int `json:"index"`
}

type kungfuConfig struct {
	Cluster clusterSpec `json:"cluster"`
	Task    taskSpec    `json:"task"`
}

func ParseConfigFromJSON(js string) (*Config, error) {
	var kfConfig kungfuConfig
	if err := json.Unmarshal([]byte(js), &kfConfig); err != nil {
		return nil, err
	}
	var initPeers plan.PeerList
	for _, p := range kfConfig.Cluster.Worker {
		initPeers = append(initPeers, plan.PeerID(p))
	}
	return &Config{
		InitPeers: initPeers,
		// Strategy:  *strategy,
	}, nil
}
