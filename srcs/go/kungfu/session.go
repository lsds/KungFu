package kungfu

import (
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type partition struct {
	Graphs []*plan.Graph
}

type session struct {
	partitions []partition
}

type partitionStrategy func([]plan.PeerSpec) []partition

var partitionStrategies = map[kb.KungFu_AllReduceAlgo]partitionStrategy{
	kb.KungFu_Star:   createStarPartitions,
	kb.KungFu_Clique: createCliquePartitions,
	kb.KungFu_Ring:   createRingPartitions,
	kb.KungFu_Tree:   createTreePartitions,
}

func newSession(c Config, ps *plan.ProcSpec) *session {
	f := partitionStrategies[c.Algo]
	if f == nil {
		log.Warnf("%s is not implemeted, fallback to %s", c.Algo, kb.KungFu_Star)
		f = createStarPartitions
	}
	return &session{partitions: f(ps.Peers)}
}

func createStarPartitions(peers []plan.PeerSpec) []partition {
	k := len(peers)
	bcastGraph := plan.GenStarBcastGraph(k, 0)
	gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
	return []partition{
		{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		},
	}
}

func createTreePartitions(peers []plan.PeerSpec) []partition {
	bcastGraph := plan.GenDefaultBcastGraph(peers)
	gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
	return []partition{
		{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		},
	}
}

func createCliquePartitions(peers []plan.PeerSpec) []partition {
	k := len(peers)
	var ps []partition
	for r := 0; r < k; r++ {
		bcastGraph := plan.GenStarBcastGraph(k, r)
		gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
		ps = append(ps, partition{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		})
	}
	return ps
}

func createRingPartitions(peers []plan.PeerSpec) []partition {
	k := len(peers)
	var ps []partition
	for r := 0; r < k; r++ {
		gatherGraph, bcastGraph := plan.GenCircularGraphPair(k, r)
		ps = append(ps, partition{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		})
	}
	return ps
}
