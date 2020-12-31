package env

import (
	"os"
	"strconv"

	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

// ParseConfigFromOpenMPIEnv is for debug only
func ParseConfigFromOpenMPIEnv() (*Config, error) {
	mpiSize, err := strconv.Atoi(os.Getenv(`OMPI_COMM_WORLD_SIZE`))
	if err != nil {
		return nil, err
	}
	mpiRank, err := strconv.Atoi(os.Getenv(`OMPI_COMM_WORLD_RANK`))
	if err != nil {
		return nil, err
	}
	hl := plan.HostList{{
		IPv4:  plan.MustParseIPv4(`127.0.0.1`),
		Slots: mpiSize,
	}}
	peers, err := hl.GenPeerList(mpiSize, plan.DefaultPortRange)
	assert.OK(err)
	return &Config{
		Self:      peers[mpiRank],
		InitPeers: peers,
		Strategy:  base.DefaultStrategy,
	}, nil
}
