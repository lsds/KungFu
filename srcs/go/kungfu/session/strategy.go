package session

import (
	"bytes"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/graph"
	"github.com/lsds/KungFu/srcs/go/plan/subgraph"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

type strategyList []strategy

type partitionStrategy func(plan.PeerList) strategyList

func (sl strategyList) choose(i int) strategy {
	return sl[i%len(sl)]
}

func (sl strategyList) digestBytes() []byte {
	b := &bytes.Buffer{}
	for _, s := range sl {
		b.Write(s.reduceGraph.DigestBytes())
		b.Write(s.bcastGraph.DigestBytes())
	}
	return b.Bytes()
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

func simpleStrategy(bcastGraph *graph.Graph) strategy {
	return strategy{
		reduceGraph: plan.GenDefaultReduceGraph(bcastGraph),
		bcastGraph:  bcastGraph,
	}
}

func createStarStrategies(peers plan.PeerList) strategyList {
	bcastGraph := plan.GenStarBcastGraph(len(peers), defaultRoot)
	return strategyList{simpleStrategy(bcastGraph)}
}

func createTreeStrategies(peers plan.PeerList) strategyList {
	bcastGraph := plan.GenTree(peers)
	return strategyList{simpleStrategy(bcastGraph)}
}

func createBinaryTreeStrategies(peers plan.PeerList) strategyList {
	bcastGraph := plan.GenBinaryTree(len(peers))
	return strategyList{simpleStrategy(bcastGraph)}
}

func createBinaryTreeStarStrategies(peers plan.PeerList) strategyList {
	bcastGraph := plan.GenBinaryTreeStar(peers)
	return strategyList{simpleStrategy(bcastGraph)}
}

func createMultiBinaryTreeStarStrategies(peers plan.PeerList) strategyList {
	var sl strategyList
	for _, bcastGraph := range plan.GenMultiBinaryTreeStar(peers) {
		sl = append(sl, simpleStrategy(bcastGraph))
	}
	return sl
}

func createCliqueStrategies(peers plan.PeerList) strategyList {
	k := len(peers)
	var sl strategyList
	for r := 0; r < k; r++ {
		bcastGraph := plan.GenStarBcastGraph(k, r)
		sl = append(sl, simpleStrategy(bcastGraph))
	}
	return sl
}

func createRingStrategies(peers plan.PeerList) strategyList {
	k := len(peers)
	var sl strategyList
	for r := 0; r < k; r++ {
		reduceGraph, bcastGraph := plan.GenCircularGraphPair(k, r)
		sl = append(sl, strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		})
	}
	return sl
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

func genLocalStrategyList(peers plan.PeerList) strategyList {
	masters, parents := peers.PartitionByHost()
	bcastGraph, m, ok := graph.FromForestArray(parents)
	assert.True(ok)
	assert.True(m == len(masters))
	return strategyList{simpleStrategy(bcastGraph)}
}

func genGlobalStrategyList(peers plan.PeerList, strategyName kb.Strategy) strategyList {
	return partitionStrategies[strategyName](peers)
}

func createCrossRingStrategies(peers plan.PeerList) strategyList {
	n := len(peers)
	masters, _ := peers.PartitionByHost()
	var sl strategyList
	for r := range masters {
		reduceGraph, bcastGraph := subgraph.GenCircularGraphPair(n, masters, r)
		sl = append(sl, strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		})
	}
	return sl
}

func createCrossBinaryTreeStrategies(peers plan.PeerList) strategyList {
	masters, _ := peers.PartitionByHost()
	bcastGraph := subgraph.GenBinaryTree(len(peers), masters)
	return strategyList{simpleStrategy(bcastGraph)}
}

func genCrossStrategyList(peers plan.PeerList, strategyName kb.Strategy) strategyList {
	if strategyName == kb.Ring {
		return createCrossRingStrategies(peers)
	}
	return createCrossBinaryTreeStrategies(peers)
}
