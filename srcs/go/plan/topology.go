package plan

import "sort"

func GenDefaultBcastGraph(peers []PeerSpec) *Graph {
	g := NewGraph(len(peers))
	hostMasters := make(map[string]int)
	for rank, p := range peers {
		if master, ok := hostMasters[p.NetAddr.Host]; !ok {
			hostMasters[p.NetAddr.Host] = rank
		} else {
			g.AddEdge(master, rank)
		}
	}
	var masters []int
	for _, rank := range hostMasters {
		masters = append(masters, rank)
	}
	sort.Ints(masters)
	if len(masters) > 1 {
		for _, rank := range masters[1:] {
			g.AddEdge(masters[0], rank)
		}
	}
	return g
}

func GenDefaultReduceGraph(g *Graph) *Graph {
	g0 := g.Reverse()
	k := len(g.Nodes)
	for i := 0; i < k; i++ {
		g0.AddEdge(i, i)
	}
	return g0
}

// GenStarBcastGraph generates a star shape graph with k vertices and centered at vertice r (0 <= r < k)
func GenStarBcastGraph(k, r int) *Graph {
	g := NewGraph(k)
	for i := 0; i < k; i++ {
		if i != r {
			g.AddEdge(r, i)
		}
	}
	return g
}

func GenCircularGraphPair(k, r int) (*Graph, *Graph) {
	g := NewGraph(k)
	for i := 0; i < k; i++ {
		g.AddEdge(i, i)
	}
	b := NewGraph(k)
	for i := 1; i < k; i++ {
		g.AddEdge((r+i)%k, (r+i+1)%k)
		b.AddEdge((r+i-1)%k, (r+i)%k)
	}
	return g, b
}
