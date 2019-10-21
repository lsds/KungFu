package modelarts

import (
	"errors"
	"fmt"
	"net"
	"os"
	"strconv"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

const (
	TaskIndexEnvKey = `DLS_TASK_INDEX`
	TaskNumEnvKey   = `DLS_TASK_NUMBER`
	PeerAddrFormat  = `BATCH_CUSTOM%d_HOSTS`
)

type ContainerInfo struct {
	Self     plan.PeerID
	PeerList plan.PeerList
}

func ParseEnv() (*ContainerInfo, error) {
	idx, err := requireInt(TaskIndexEnvKey)
	if err != nil {
		return nil, err
	}
	num, err := requireInt(TaskNumEnvKey)
	if err != nil {
		return nil, err
	}
	pl, err := parsePeerList(num)
	if err != nil {
		return nil, err
	}
	log.Infof("cluster size: %d, idx: %d", num, idx)
	if idx < 0 || num <= idx {
		log.Warnf("invalid idx, 0 <= idx < cluster size is required")
	}
	if num == 1 && idx == 1 {
		log.Warnf("changing idx=1 to idx=0 when cluster size=1")
		idx = 0
	}
	return &ContainerInfo{
		Self:     pl[idx],
		PeerList: pl,
	}, nil
}

func requireInt(key string) (int, error) {
	val := os.Getenv(key)
	if len(val) <= 0 {
		return 0, fmt.Errorf("%s not set", key)
	}
	n, err := strconv.Atoi(val)
	if err != nil {
		return 0, err
	}
	return n, nil
}

func parsePeerList(n int) (plan.PeerList, error) {
	if n == 1 {
		peer := plan.PeerID{
			IPv4: plan.MustParseIPv4("127.0.0.1"),
			Port: uint16(38888),
		}
		return []plan.PeerID{peer}, nil
	}

	var peers []plan.PeerID
	for i := 0; i < n; i++ {
		key := fmt.Sprintf(PeerAddrFormat, i)
		val := os.Getenv(key)
		if len(val) <= 0 {
			return nil, fmt.Errorf("%s not set", key)
		}
		ipv4, port, err := resolvePeer(val)
		if err != nil {
			return nil, err
		}
		peer := plan.PeerID{
			IPv4: ipv4,
			Port: uint16(port),
		}
		log.Infof("peer %d %s resolved as %s", i, val, peer)
		peers = append(peers, peer)
	}
	return peers, nil
}

func resolvePeer(hostPort string) (uint32, int, error) {
	h, p, err := net.SplitHostPort(hostPort)
	if err != nil {
		return 0, 0, err
	}
	addrs, err := net.LookupHost(h)
	if err != nil {
		return 0, 0, err
	}
	if len(addrs) != 1 {
		return 0, 0, errors.New("exactly 1 addr is expected")
	}
	port, err := strconv.Atoi(p)
	if len(addrs) != 1 {
		return 0, 0, err
	}
	ipv4, err := plan.ParseIPv4(addrs[0])
	if err != nil {
		return 0, 0, err
	}
	return ipv4, port, nil
}
