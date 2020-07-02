package subgraph

import "github.com/lsds/KungFu/srcs/go/plan/graph"

func GenCircularGraphPair(n int, vs []int, r int) (*graph.Graph, *graph.Graph) {
	rg := graph.New(n)
	bg := graph.New(n)
	k := len(vs)
	for i := 0; i < k; i++ {
		rg.AddEdge(vs[i], vs[i])
	}
	for i := 1; i < k; i++ {
		rg.AddEdge(vs[(r+i)%k], vs[(r+i+1)%k])
		bg.AddEdge(vs[(r+i-1)%k], vs[(r+i)%k])
	}
	return rg, bg
}

func GenBinaryTree(n int, vs []int) *graph.Graph {
	g := graph.New(n)
	k := len(vs)
	for i := 0; i < k; i++ {
		if j := i*2 + 1; j < k {
			g.AddEdge(vs[i], vs[j])
		}
		if j := i*2 + 2; j < k {
			g.AddEdge(vs[i], vs[j])
		}
	}
	return g
}
