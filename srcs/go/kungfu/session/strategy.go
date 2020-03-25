package session

import (
	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type partitionStrategy func(plan.PeerList) []strategy

type strategyList []strategy

func (sl strategyList) choose(i int) strategy {
	return sl[i%len(sl)]
}

var partitionStrategies = map[kb.Strategy]partitionStrategy{
	kb.Star:                createStarStrategies,
	kb.Clique:              createCliqueStrategies,
	kb.Ring:                createRingStrategies,
	kb.Tree:                createTreeStrategies,
	kb.BinaryTree:          createBinaryTreeStrategies,
	kb.BinaryTreeStar:      createBinaryTreeStarStrategies,
	kb.MultiBinaryTreeStar: createMultiBinaryTreeStarStrategies,
}

func simpleSingleGraphStrategy(bcastGraph *plan.Graph) []strategy {
	return []strategy{
		{
			reduceGraph: plan.GenDefaultReduceGraph(bcastGraph),
			bcastGraph:  bcastGraph,
		},
	}
}

func createStarStrategies(peers plan.PeerList) []strategy {
	bcastGraph := plan.GenStarBcastGraph(len(peers), defaultRoot)
	return simpleSingleGraphStrategy(bcastGraph)
}

func createTreeStrategies(peers plan.PeerList) []strategy {
	bcastGraph := plan.GenTree(peers)
	return simpleSingleGraphStrategy(bcastGraph)
}

func createBinaryTreeStrategies(peers plan.PeerList) []strategy {
	bcastGraph := plan.GenBinaryTree(len(peers))
	return simpleSingleGraphStrategy(bcastGraph)
}

func createBinaryTreeStarStrategies(peers plan.PeerList) []strategy {
	bcastGraph := plan.GenBinaryTreeStar(peers)
	return simpleSingleGraphStrategy(bcastGraph)
}

func createMultiBinaryTreeStarStrategies(peers plan.PeerList) []strategy {
	var ss []strategy
	for _, bcastGraph := range plan.GenMultiBinaryTreeStar(peers) {
		ss = append(ss, strategy{
			reduceGraph: plan.GenDefaultReduceGraph(bcastGraph),
			bcastGraph:  bcastGraph,
		})
	}
	return ss
}

func createCliqueStrategies(peers plan.PeerList) []strategy {
	k := len(peers)
	var ss []strategy
	for r := 0; r < k; r++ {
		bcastGraph := plan.GenStarBcastGraph(k, r)
		reduceGraph := plan.GenDefaultReduceGraph(bcastGraph)
		ss = append(ss, strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		})
	}
	return ss
}

func createRingStrategies(peers plan.PeerList) []strategy {
	k := len(peers)
	var ss []strategy
	for r := 0; r < k; r++ {
		reduceGraph, bcastGraph := plan.GenCircularGraphPair(k, r)
		ss = append(ss, strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		})
	}
	return ss
}

func autoSelect(peers plan.PeerList) kb.Strategy {
	m := make(map[uint32]int)
	for _, p := range peers {
		m[p.IPv4]++
	}
	if len(m) == 1 {
		return kb.Star
	}
	return kb.BinaryTreeStar
}
