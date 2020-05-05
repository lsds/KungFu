package plan

import (
	"bytes"
	"fmt"
)

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

func NewGraphFromTreeArray(tree []int32) *Graph {
	g := NewGraph(len(tree))
	for i, father := range tree {
		if int32(i) != father {
			g.AddEdge(int(father), int(i))
		}
	}
	return g
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

func (g *Graph) DebugString() string {
	b := &bytes.Buffer{}
	fmt.Fprintf(b, "[%d]{", len(g.Nodes))
	for i := range g.Nodes {
		if g.IsSelfLoop(i) {
			fmt.Fprintf(b, "(%d)", i)
		}
	}
	for i, n := range g.Nodes {
		for _, j := range n.Nexts {
			fmt.Fprintf(b, "(%d->%d)", i, j)
		}
	}
	fmt.Fprintf(b, "}")
	return b.String()
}

func (g *Graph) Debug() {
	fmt.Printf("%s", g.DebugString())
}
