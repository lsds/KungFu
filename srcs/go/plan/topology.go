package plan

func getLocalMasters(peers PeerList) ([]int, map[uint32]int) {
	var masters []int
	hostMaster := make(map[uint32]int)
	for rank, p := range peers {
		if _, ok := hostMaster[p.IPv4]; !ok {
			hostMaster[p.IPv4] = rank
			masters = append(masters, rank)
		}
	}
	return masters, hostMaster
}

func GenDefaultBcastGraph(peers PeerList) *Graph {
	g := NewGraph(len(peers))
	masters, hostMaster := getLocalMasters(peers)
	for rank, p := range peers {
		if master := hostMaster[p.IPv4]; master != rank {
			g.AddEdge(master, rank)
		}
	}
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

func GenBinaryTree(k int) *Graph {
	g := NewGraph(k)
	for i := 0; i < k; i++ {
		if j := i*2 + 1; j < k {
			g.AddEdge(i, j)
		}
		if j := i*2 + 2; j < k {
			g.AddEdge(i, j)
		}
	}
	return g
}

func GenBinaryTreeStar(peers PeerList) *Graph {
	g := NewGraph(len(peers))
	masters, hostMaster := getLocalMasters(peers)
	for rank, p := range peers {
		if master := hostMaster[p.IPv4]; master != rank {
			g.AddEdge(master, rank)
		}
	}
	if k := len(masters); k > 1 {
		for i := 0; i < k; i++ {
			if j := i*2 + 1; j < k {
				g.AddEdge(masters[i], masters[j])
			}
			if j := i*2 + 2; j < k {
				g.AddEdge(masters[i], masters[j])
			}
		}
	}
	return g
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
