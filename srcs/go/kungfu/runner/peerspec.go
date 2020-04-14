package runner

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

type PeerSpec struct {
	Host string
	Port uint16
	Slot int
}

func (p PeerSpec) String() string {
	return fmt.Sprintf("%s:%d:%d", p.Host, p.Port, p.Slot)
}

var errInvalidPeerSpec = errors.New("invalid peer spec")

func ParsePeerSpec(config string) (*PeerSpec, error) {
	parts := strings.Split(config, ":")
	if len(parts) < 1 {
		return nil, errInvalidPeerSpec
	}
	host := parts[0]
	if len(parts) < 2 {
		return nil, errInvalidPeerSpec
	}
	port, err := strconv.Atoi(parts[1])
	if err != nil {
		return nil, errInvalidPeerSpec
	}
	if int(uint16(port)) != port {
		return nil, errInvalidPeerSpec
	}
	var slot int
	if len(parts) > 2 {
		n, err := strconv.Atoi(parts[2])
		if err != nil {
			return nil, errInvalidPeerSpec
		}
		slot = n
	}
	return &PeerSpec{
		Host: host,
		Port: uint16(port),
		Slot: slot,
	}, nil
}

type PeerSpecList []PeerSpec

func (l PeerSpecList) String() string {
	var parts []string
	for _, p := range l {
		parts = append(parts, p.String())
	}
	return strings.Join(parts, ",")
}

var errInvalidPeerSpecList = errors.New("invalid peer spec list")

func ParsePeerSpecList(config string) (PeerSpecList, error) {
	var l PeerSpecList
	if len(config) == 0 {
		return l, nil
	}
	parts := strings.Split(config, ",")
	m1 := make(map[string]struct{})
	m2 := make(map[string]struct{})
	for _, part := range parts {
		p, err := ParsePeerSpec(part)
		if err != nil {
			return nil, errInvalidPeerSpecList
		}
		k1 := fmt.Sprintf("%s:%d", p.Host, p.Port)
		k2 := fmt.Sprintf("%s:%d", p.Host, p.Slot)
		if _, ok := m1[k1]; ok {
			return nil, errInvalidPeerSpecList
		}
		m1[k1] = struct{}{}
		if _, ok := m2[k2]; ok {
			if p.Slot > 0 {
				return nil, errInvalidPeerSpecList
			}
		}
		m2[k2] = struct{}{}
		l = append(l, *p)
	}
	return l, nil
}

func (l PeerSpecList) Eq(r PeerSpecList) bool {
	if len(l) != len(r) {
		return false
	}
	for i, p := range l {
		if p != r[i] {
			return false
		}
	}
	return true
}
