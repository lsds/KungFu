package session

import (
	"bytes"
	"fmt"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/graph"
	"github.com/lsds/KungFu/srcs/go/plan/subgraph"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

type strategyList []strategy

type partitionStrategy func(plan.PeerList) strategyList

// A strategy is a pair of graphs for collective communication
type strategy struct {
	reduceGraph *graph.Graph
	bcastGraph  *graph.Graph
	stat        *StrategyStat
}

func newStrategy(rg *graph.Graph, bg *graph.Graph) strategy {
	return strategy{
		reduceGraph: rg,
		bcastGraph:  bg,
		stat:        &StrategyStat{},
	}
}

func (sl strategyList) choose(i int) strategy {

	idx := i % len(sl)
	j := idx
	for sl[idx].stat.suspended {
		i++
		idx = i % len(sl)
		if j == idx {
			//TODO: fix this
			//This shouldn't happen
			panic("All strategies were suspended")
		}
	}

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
	kb.MultiStar:           createMultiStarStrategies,
	kb.Clique:              createCliqueStrategies,
	kb.Ring:                createRingStrategies,
	kb.Tree:                createTreeStrategies,
	kb.BinaryTree:          createBinaryTreeStrategies,
	kb.BinaryTreeStar:      createBinaryTreeStarStrategies,
	kb.MultiBinaryTreeStar: createMultiBinaryTreeStarStrategies,
}

func simpleStrategy(bcastGraph *graph.Graph) strategy {
	return newStrategy(plan.GenDefaultReduceGraph(bcastGraph), bcastGraph)
}

func simpleSingleGraphStrategy(bcastGraph *graph.Graph) strategyList {
	return strategyList{simpleStrategy(bcastGraph)}
}

func createStarStrategies(peers plan.PeerList) strategyList {
	bcastGraph := plan.GenStarBcastGraph(len(peers), defaultRoot)
	// fmt.Println("Printing strategy #1")
	// fmt.Println("Bcast Tree:")
	// bcastGraph.Debug()
	return simpleSingleGraphStrategy(bcastGraph)
}

func createMultiStarStrategies(peers plan.PeerList) strategyList {
	var ss []strategy
	//fmt.Println("DEV::createMultiStarStrategies:: Going to print generated trees")
	for _, bcastGraph := range plan.GenMultiStar(peers) {
		ss = append(ss, simpleStrategy(bcastGraph))
		// fmt.Println("Printing strategy #", i)
		// fmt.Println("Bcast Tree:")
		// ss[len(ss)-1].bcastGraph.Debug()
		// fmt.Println("\nReduce Tree:")
		// ss[len(ss)-1].reduceGraph.Debug()
	}
	// fmt.Println("DEV::creatingMultipleStarStrategy:: created ", len(ss), "different strategies")
	return ss
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
	//TODO: remove printing, just debug purpose
	fmt.Println("DEV::createMultiBinaryTreeStarStrategies:: Going to print generated trees")
	for i, bcastGraph := range plan.GenMultiBinaryTreeStar(peers) {
		sl = append(sl, simpleStrategy(bcastGraph))
		fmt.Println("Printing strategy #", i)
		fmt.Println("Bcast Tree:")
		sl[len(sl)-1].bcastGraph.Debug()
		fmt.Println("\nReduce Tree:")
		sl[len(sl)-1].reduceGraph.Debug()
	}
	fmt.Println("DEV::creatingMultipleBinaryTreeStarStrategy:: created ", len(sl), "different strategies")
	return sl
}

func createCliqueStrategies(peers plan.PeerList) strategyList {
	k := len(peers)
	var sl strategyList
	for r := 0; r < k; r++ {
		bcastGraph := plan.GenStarBcastGraph(k, r)
		reduceGraph := plan.GenDefaultReduceGraph(bcastGraph)
		sl = append(sl, newStrategy(reduceGraph, bcastGraph))
	}
	return sl
}

func createRingStrategies(peers plan.PeerList) strategyList {
	k := len(peers)
	var sl strategyList
	for r := 0; r < k; r++ {
		reduceGraph, bcastGraph := plan.GenCircularGraphPair(k, r)
		sl = append(sl, newStrategy(reduceGraph, bcastGraph))
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
		sl = append(sl, newStrategy(reduceGraph, bcastGraph))
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
