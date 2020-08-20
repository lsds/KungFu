package graph

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"sort"
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

func (n *Node) isIsolated() bool {
	return len(n.Prevs) == 0 && len(n.Nexts) == 0
}

// Graph represents a graph of integers numbered from 0 to n - 1.
type Graph struct {
	Nodes []Node
}

func New(n int) *Graph {
	var nodes []Node
	for i := 0; i < n; i++ {
		nodes = append(nodes, Node{Rank: i})
	}
	return &Graph{
		Nodes: nodes,
	}
}

// FromForestArray creates a Graph from array representation of a forest
// f[i] represents the father of i, if f[i] != i
func FromForestArray(forest []int) (*Graph, int, bool) {
	var m int
	n := len(forest)
	g := New(n)
	for i, father := range forest {
		switch {
		case father < 0 || father >= n:
			return nil, 0, false
		case father == i:
			m++
		default:
			g.AddEdge(father, i)
		}
	}
	// FIXME: check cycle!
	return g, m, true
}

func FromForestArrayI32(forest []int32) (*Graph, int, bool) {
	f := make([]int, len(forest))
	for i, r := range forest {
		f[i] = int(r)
	}
	return FromForestArray(f)
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

func (g Graph) IsIsolated(i int) bool {
	return g.Nodes[i].isIsolated()
}

func (g Graph) Prevs(i int) []int {
	return g.Nodes[i].Prevs
}

func (g Graph) Nexts(i int) []int {
	return g.Nodes[i].Nexts
}

func (g Graph) Reverse() *Graph {
	r := New(len(g.Nodes))
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

func (g *Graph) DigestBytes() []byte {
	b := &bytes.Buffer{}
	w32 := func(x int32) { binary.Write(b, binary.LittleEndian, x) }
	w32(int32(len(g.Nodes)))
	for _, node := range g.Nodes {
		deg := len(node.Nexts)
		vs := make([]int, deg)
		copy(vs, node.Nexts)
		sort.Ints(vs)
		w32(b2i(node.SelfLoop))
		w32(int32(deg))
		for _, j := range vs {
			w32(int32(j))
		}
	}
	return b.Bytes()
}

func b2i(b bool) int32 {
	if b {
		return 1
	}
	return 0
}
