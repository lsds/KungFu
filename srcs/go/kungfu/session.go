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

func newSession(c Config, ps *plan.ProcSpec) *session {
	k := ps.Size()
	partitions := func() []partition {
		switch c.Algo {
		case kb.KungFu_Tree:
			return treeGraphs(ps.Peers)
		case kb.KungFu_Clique:
			return cliqueGraphs(k)
		case kb.KungFu_Ring:
			return ringGraphs(k)
		case kb.KungFu_Simple:
			return simplePartitions(k)
		default:
			log.Warnf("%s is not implemeted, fallback to simpleAllReduce", c.Algo)
			return simplePartitions(k)
		}
	}()
	return &session{partitions: partitions}
}

func simplePartitions(k int) []partition {
	bcastGraph := plan.GenStarBcastGraph(k, 0)
	gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
	return []partition{
		{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		},
	}
}

func treeGraphs(tasks []plan.TaskSpec) []partition {
	bcastGraph := plan.GenDefaultBcastGraph(tasks)
	gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
	return []partition{
		{
			Graphs: []*plan.Graph{gatherGraph, bcastGraph},
		},
	}
}

func cliqueGraphs(k int) []partition {
	var ps []partition
	for r := 0; r < k; r++ {
		bcastGraph := plan.GenStarBcastGraph(k, r)
		gatherGraph := plan.GenDefaultGatherGraph(bcastGraph)
		ps = append(ps, partition{
			Graphs: []*plan.Graph{
				gatherGraph,
				bcastGraph,
			},
		})
	}
	return ps
}

func ringGraphs(k int) []partition {
	var ps []partition
	for r := 0; r < k; r++ {
		gatherGraph, bcastGraph := plan.GenCircularGraphPair(k, r)
		ps = append(ps, partition{
			Graphs: []*plan.Graph{
				gatherGraph,
				bcastGraph,
			},
		})
	}
	return ps
}
