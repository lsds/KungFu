package rchannel

import "fmt"

// FIXME: make members private, public is required by JSON encoding for now

type Node struct {
	Rank  int
	Prevs []int
	Nexts []int
}

type Graph struct {
	Nodes []Node
}

func newGraph(n int) *Graph {
	var nodes []Node
	for i := 0; i < n; i++ {
		nodes = append(nodes, Node{Rank: i})
	}
	return &Graph{
		Nodes: nodes,
	}
}

func (g *Graph) AddEdge(i, j int) {
	g.Nodes[i].Nexts = append(g.Nodes[i].Nexts, j)
	g.Nodes[j].Prevs = append(g.Nodes[j].Prevs, i)
}

func (g *Graph) Prevs(i int) []int {
	return g.Nodes[i].Prevs
}

func (g *Graph) Nexts(i int) []int {
	return g.Nodes[i].Nexts
}

func (g *Graph) Debug() {
	fmt.Print("graph {\n")
	for i, n := range g.Nodes {
		fmt.Printf("\t%d;\n", n.Rank)
		for _, j := range n.Nexts {
			fmt.Printf("\t%d -> %d;\n", i, j)
		}
	}
	fmt.Print("}\n")
}
