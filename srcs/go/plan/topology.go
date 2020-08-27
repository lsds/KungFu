package plan

import "github.com/lsds/KungFu/srcs/go/plan/graph"

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

func GenTree(peers PeerList) *graph.Graph {
	g := graph.New(len(peers))
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

func GenDefaultReduceGraph(g *graph.Graph) *graph.Graph {
	g0 := g.Reverse()
	k := len(g.Nodes)
	for i := 0; i < k; i++ {
		g0.AddEdge(i, i)
	}
	return g0
}

func GenBinaryTree(k int) *graph.Graph {
	g := graph.New(k)
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

func genMultiStar(peers PeerList, root int) *graph.Graph {
	g := graph.New(len(peers))
	masters, hostMaster := getLocalMasters(peers)
	//create star topology in each different machine
	for rank, p := range peers {
		if master := hostMaster[p.IPv4]; master != rank {
			g.AddEdge(master, rank)
		}
	}
	//create star topology between different machines
	if k := len(masters); k > 1 {
		for i := 0; i < k; i++ {
			if i != root {
				g.AddEdge(masters[root], masters[i])
			}
		}
	}

	return g
}

func genBinaryTreeStar(peers PeerList, offset int) *graph.Graph {
	g := graph.New(len(peers))
	masters, hostMaster := getLocalMasters(peers)
	//create star topology in each different machine
	for rank, p := range peers {
		if master := hostMaster[p.IPv4]; master != rank {
			g.AddEdge(master, rank)
		}
	}
	//create the tree between different machines
	if k := len(masters); k > 1 {
		idx := func(i int) int {
			return (i + offset) % k
		}
		for i := 0; i < k; i++ {
			if j := i*2 + 1; j < k {
				g.AddEdge(masters[idx(i)], masters[idx(j)])
			}
			if j := i*2 + 2; j < k {
				g.AddEdge(masters[idx(i)], masters[idx(j)])
			}
		}
	}

	return g
}

func GenBinaryTreeStar(peers PeerList) *graph.Graph {
	return genBinaryTreeStar(peers, 0)
}

func GenMultiBinaryTreeStar(peers PeerList) []*graph.Graph {
	var gs []*graph.Graph
	masters, _ := getLocalMasters(peers)
	m := len(masters)
	for i := 0; i < m; i++ {
		gs = append(gs, genBinaryTreeStar(peers, i))
	}
	return gs
}

func GenMultiStar(peers PeerList) []*graph.Graph {
	var gs []*graph.Graph
	masters, _ := getLocalMasters(peers)
	m := len(masters)
	for i := 0; i < m; i++ {
		gs = append(gs, genMultiStar(peers, i))
	}
	return gs
}

func GenAlternativeStar(peers PeerList, off int) *graph.Graph {
	var gs []*graph.Graph
	masters, _ := getLocalMasters(peers)
	m := len(masters)
	for i := 0; i < m; i++ {
		gs = append(gs, genMultiStar(peers, i))
	}
	return genMultiStar(peers, masters[off])
}

// GenStarBcastGraph generates a star shape graph with k vertices and centered at vertice r (0 <= r < k)
func GenStarBcastGraph(k, r int) *graph.Graph {
	g := graph.New(k)
	for i := 0; i < k; i++ {
		if i != r {
			g.AddEdge(r, i)
		}
	}

	return g
}

func GenCircularGraphPair(k, r int) (*graph.Graph, *graph.Graph) {
	g := graph.New(k)
	for i := 0; i < k; i++ {
		g.AddEdge(i, i)
	}
	b := graph.New(k)
	for i := 1; i < k; i++ {
		g.AddEdge((r+i)%k, (r+i+1)%k)
		b.AddEdge((r+i-1)%k, (r+i)%k)
	}
	return g, b
}
