package plan

import (
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"strings"

	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

var ErrInvalidHostSpec = errors.New("Invalid HostSpec")

type HostSpec struct {
	IPv4       uint32
	Slots      int
	PublicAddr string
}

func (h HostSpec) String() string {
	return fmt.Sprintf("%s:%d:%s", FormatIPv4(h.IPv4), h.Slots, h.PublicAddr)
}

func (h HostSpec) DebugString() string {
	return fmt.Sprintf("%s slots=%d hostname=%s", FormatIPv4(h.IPv4), h.Slots, h.PublicAddr)
}

func parseHostSpec(spec string) (*HostSpec, error) {
	parts := strings.Split(spec, ":")
	if len(parts) < 1 {
		return nil, ErrInvalidHostSpec
	}
	ipv4, err := ParseIPv4(parts[0])
	if err != nil {
		return nil, err
	}
	switch len(parts) {
	case 1:
		return &HostSpec{IPv4: ipv4, Slots: 1, PublicAddr: parts[0]}, nil
	case 2:
		slots, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, ErrInvalidHostSpec
		}
		return &HostSpec{IPv4: ipv4, Slots: slots, PublicAddr: parts[0]}, nil
	case 3:
		slots, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, ErrInvalidHostSpec
		}
		return &HostSpec{IPv4: ipv4, Slots: slots, PublicAddr: parts[2]}, nil
	}
	return nil, ErrInvalidHostSpec
}

type HostList []HostSpec

var DefaultHostList = HostList{
	{
		IPv4:       MustParseIPv4(`127.0.0.1`),
		Slots:      runtime.NumCPU(),
		PublicAddr: `127.0.0.1`,
	},
}

func (hl HostList) String() string {
	var ss []string
	for _, h := range hl {
		ss = append(ss, h.String())
	}
	return strings.Join(ss, ",")
}

func ParseHostList(hostlist string) (HostList, error) {
	var hl HostList
	if len(hostlist) == 0 {
		return hl, nil
	}
	for _, h := range strings.Split(hostlist, ",") {
		spec, err := parseHostSpec(h)
		if err != nil {
			return nil, err
		}
		hl = append(hl, *spec)
	}
	return hl, nil
}

func (hl HostList) SlotOf(ipv4 uint32) int {
	for _, h := range hl {
		if h.IPv4 == ipv4 {
			return h.Slots
		}
	}
	return 0
}

func (hl HostList) Cap() int {
	var cap int
	for _, h := range hl {
		cap += h.Slots
	}
	return cap
}

func (hl HostList) LookupHost(ipv4 uint32) string {
	for _, h := range hl {
		if h.IPv4 == ipv4 {
			return h.PublicAddr
		}
	}
	return FormatIPv4(ipv4)
}

type PortRange struct {
	Begin uint16
	End   uint16
}

var DefaultPortRange = PortRange{
	Begin: 10000,
	End:   11000,
}

const DefaultRunnerPort = uint16(38080)

var errInvalidPortRange = errors.New("invalid port range")

func ParsePortRange(val string) (*PortRange, error) {
	var begin, end uint16
	if _, err := fmt.Sscanf(val, "%d-%d", &begin, &end); err != nil {
		return nil, err
	}
	if end < begin {
		return nil, errInvalidPortRange
	}
	return &PortRange{Begin: begin, End: end}, nil
}

func (pr PortRange) Cap() int {
	return int(pr.End - pr.Begin + 1)
}

func (pr PortRange) String() string {
	return fmt.Sprintf("%d-%d", pr.Begin, pr.End)
}

// Set implements flags.Value::Set
func (pr *PortRange) Set(val string) error {
	value, err := ParsePortRange(val)
	if err != nil {
		return err
	}
	*pr = *value
	return nil
}

func (hl HostList) ShrinkToFit(np int) HostList {
	var cap int
	var part HostList
	for _, h := range hl {
		part = append(part, h)
		cap += h.Slots
		if cap >= np {
			break
		}
	}
	return part
}

func (hl HostList) genPeerList(np int, pr PortRange) PeerList {
	var pl PeerList
	if np == 0 {
		return pl
	}
	for _, host := range hl {
		for j := 0; j < host.Slots; j++ {
			id := PeerID{
				IPv4: host.IPv4,
				Port: pr.Begin + uint16(j),
			}
			pl = append(pl, id)
			if len(pl) >= np {
				return pl
			}
		}
	}
	return pl
}

var ErrNoEnoughCapacity = errors.New("no enough capacity")

func (hl HostList) GenRunnerList(port uint16) PeerList {
	var pl PeerList
	for _, h := range hl {
		pl = append(pl, PeerID{IPv4: h.IPv4, Port: port})
	}
	return pl
}

func (hl HostList) GenPeerList(np int, pr PortRange) (PeerList, error) {
	if hl.Cap() < np {
		return nil, ErrNoEnoughCapacity
	}
	for _, h := range hl {
		if pr.Cap() < h.Slots {
			return nil, ErrNoEnoughCapacity
		}
	}
	return hl.genPeerList(np, pr), nil
}

func (hl HostList) MustGenPeerList(np int, pr PortRange) PeerList {
	pl, err := hl.GenPeerList(np, pr)
	assert.OK(err)
	return pl
}
