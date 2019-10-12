package plan

import (
	"fmt"
)

func fakeHosts(n int) HostList {
	var hosts HostList
	for i := 0; i < n; i++ {
		ip := MustParseIPv4(fmt.Sprintf(`192.168.1.%d`, 11+i))
		host := HostSpec{
			IPv4:       ip,
			Slots:      4,
			PublicAddr: FormatIPv4(ip),
		}
		hosts = append(hosts, host)
	}
	return hosts
}
