package rchannel

import (
	"fmt"
	"testing"
)

func fakeHosts(n int) []HostSpec {
	var hosts []HostSpec
	for i := 0; i < n; i++ {
		ip := fmt.Sprintf(`192.168.1.%d`, 11+i)
		host := HostSpec{
			Hostname:   ip,
			Slots:      4,
			PublicAddr: ip,
		}
		hosts = append(hosts, host)
	}
	return hosts
}

func Test_graph(t *testing.T) {
	n := 4
	hosts := fakeHosts(n)

	k := n * 4
	_, g1, g2 := genTaskSpecs(k, hosts)
	g1.Debug()
	g2.Debug()
	// TODO: add tests
}
