package plan

import (
	"os"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

// PeerSpec describes the system resources that will be used by the process of the peer
type PeerSpec struct {
	DeviceID       int
	NetAddr        NetAddr
	MonitoringPort uint16
}

func (ps PeerSpec) String() string {
	return toString(ps)
}

func getSelfFromEnv() (*PeerSpec, error) {
	var ps PeerSpec
	selfSpecConfig := os.Getenv(kb.SelfSpecEnvKey)
	if err := fromString(selfSpecConfig, &ps); err != nil {
		return nil, err
	}
	return &ps, nil
}
