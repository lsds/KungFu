package kungfu

import (
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

// A strategy is a sequence of dataflow graph
type strategy struct {
	Graphs []*plan.Graph
}

type session struct {
	strategies []strategy
}

type partitionStrategy func([]plan.PeerSpec) []strategy

var partitionStrategies = map[kb.KungFu_AllReduceAlgo]partitionStrategy{
	kb.KungFu_Star:   createStarStrategies,
	kb.KungFu_Clique: createCliqueStrategies,
	kb.KungFu_Ring:   createRingStrategies,
	kb.KungFu_Tree:   createTreeStrategies,
}

func newSession(c Config, ps *plan.ProcSpec) *session {
	f := partitionStrategies[c.Algo]
	if f == nil {
		log.Warnf("%s is not implemeted, fallback to %s", c.Algo, kb.KungFu_Star)
		f = createStarStrategies
	}
	return &session{strategies: f(ps.Peers)}
}

func createStarStrategies(peers []plan.PeerSpec) []strategy {
	k := len(peers)
	bcastGraph := plan.GenStarBcastGraph(k, 0)
	gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
	return []strategy{
		{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		},
	}
}

func createTreeStrategies(peers []plan.PeerSpec) []strategy {
	bcastGraph := plan.GenDefaultBcastGraph(peers)
	gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
	return []strategy{
		{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		},
	}
}

func createCliqueStrategies(peers []plan.PeerSpec) []strategy {
	k := len(peers)
	var ss []strategy
	for r := 0; r < k; r++ {
		bcastGraph := plan.GenStarBcastGraph(k, r)
		gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
		ss = append(ss, strategy{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		})
	}
	return ss
}

func createRingStrategies(peers []plan.PeerSpec) []strategy {
	k := len(peers)
	var ss []strategy
	for r := 0; r < k; r++ {
		gatherGraph, bcastGraph := plan.GenCircularGraphPair(k, r)
		ss = append(ss, strategy{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		})
	}
	return ss
}
