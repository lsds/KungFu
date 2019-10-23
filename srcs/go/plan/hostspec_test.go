package plan

import (
	"fmt"
	"testing"
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

func Test_genPeerList(t *testing.T) {
	hl := fakeHosts(1)
	pl, err := hl.GenPeerList(0, DefaultPortRange)
	if err != nil {
		t.Errorf("unexpect error: %v", err)
	}
	if n := len(pl); n != 0 {
		t.Errorf("expect %d, got %d", 0, n)
	}
}

func Test_HostSpecList(t *testing.T) {
	hl, err := ParseHostList("")
	if err != nil {
		t.Errorf("unexpect error: %v", err)
	}
	if n := len(hl); n != 0 {
		t.Errorf("expect %d, got %d", 0, n)
	}
}
