package session

import (
	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/log"
)

// A strategyHashFunc is to map a given Workspace to a communication strategy.
// KungFu can create multiple communication strategies to balance the workload in the network core and edge links.
// We support hash the Workspaces based on their names or their partition IDs if they are chunked.
type strategyHashFunc func(int, string) uint64

func simpleHash(i int, name string) uint64 {
	return uint64(i)
}

func nameBasedHash(i int, name string) uint64 {
	var h uint64
	for _, c := range name {
		h += uint64(c) * uint64(c)
	}
	return h
}

func getStrategyHash() strategyHashFunc {
	if config.StrategyHashMethod == `NAME` {
		log.Debugf("using name based hash")
		return nameBasedHash
	}
	return simpleHash
}
