package plan

import "fmt"

// FIXME: make members private, public is required by JSON encoding for now

type Vertices []int

func (vs *Vertices) Append(v int) {
	*vs = append(*vs, v)
}

type Node struct {
	Rank     int
	SelfLoop bool
	Prevs    Vertices
	Nexts    Vertices
}

type Graph struct {
	Nodes []Node
}

func NewGraph(n int) *Graph {
	var nodes []Node
	for i := 0; i < n; i++ {
		nodes = append(nodes, Node{Rank: i})
	}
	return &Graph{
		Nodes: nodes,
	}
}

func (g *Graph) AddEdge(i, j int) {
	if i == j {
		g.Nodes[i].SelfLoop = true
		return
	}
	g.Nodes[i].Nexts.Append(j)
	g.Nodes[j].Prevs.Append(i)
}

func (g Graph) IsSelfLoop(i int) bool {
	return g.Nodes[i].SelfLoop
}

func (g Graph) Prevs(i int) []int {
	return g.Nodes[i].Prevs
}

func (g Graph) Nexts(i int) []int {
	return g.Nodes[i].Nexts
}

func (g Graph) Reverse() *Graph {
	r := NewGraph(len(g.Nodes))
	for i, n := range g.Nodes {
		for _, j := range n.Nexts {
			r.Nodes[j].Nexts.Append(i)
		}
		for _, j := range n.Prevs {
			r.Nodes[j].Prevs.Append(i)
		}
	}
	return r
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
