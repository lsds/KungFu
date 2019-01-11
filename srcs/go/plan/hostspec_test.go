package plan

import (
	"fmt"
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
