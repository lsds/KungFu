package rchannel

import (
	"fmt"
	"testing"
)

func fakeHosts(n int) []string {
	var hosts []string
	for i := 0; i < n; i++ {
		hosts = append(hosts, fmt.Sprintf(`192.168.1.%d`, 11+i))
	}
	return hosts
}

func Test_graph(t *testing.T) {
	n := 4
	hosts := fakeHosts(n)

	m := 4
	k := n * m
	_, g1, g2 := genTaskSpecs(k, hosts, m)
	g1.Debug()
	g2.Debug()
	// TODO: add tests
}
