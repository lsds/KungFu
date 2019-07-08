package plan

import "testing"

func Test_graph(t *testing.T) {
	n := 4
	hosts := fakeHosts(n)

	k := n * 4
	peers := genPeerSpecs(k, hosts)

	bcastGraph := GenDefaultBcastGraph(peers)
	reduceGraph := GenDefaultreduceGraph(bcastGraph)

	reduceGraph.Debug()
	bcastGraph.Debug()
	// TODO: add tests
}
