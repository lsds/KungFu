package plan

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
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

func (hl HostList) Cap() int {
	var cap int
	for _, h := range hl {
		cap += h.Slots
	}
	return cap
}

type PortRange struct {
	Begin uint16
	End   uint16
}

var DefaultPortRange = PortRange{
	Begin: 10000,
	End:   11000,
}

var errInvalidPortRange = errors.New("invalid port range")

func getPortRangeFromEnv() (*PortRange, error) {
	val, ok := os.LookupEnv(kb.PortRangeEnvKey)
	if !ok {
		return nil, fmt.Errorf("%s not set", kb.PortRangeEnvKey)
	}
	return ParsePortRange(val)
}

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

var errNoEnoughCapacity = errors.New("no enough capacity")

func (hl HostList) GenPeerList(np int, pr PortRange) (PeerList, error) {
	if hl.Cap() < np {
		return nil, errNoEnoughCapacity
	}
	for _, h := range hl {
		if pr.Cap() < h.Slots {
			return nil, errNoEnoughCapacity
		}
	}
	return hl.genPeerList(np, pr), nil
}

func getHostListFromEnv() (HostList, error) {
	val, ok := os.LookupEnv(kb.HostListEnvKey)
	if !ok {
		return nil, fmt.Errorf("%s not set", kb.HostListEnvKey)
	}
	return ParseHostList(val)
}

func (hl HostList) GenPeerListFromFile(url string) (PeerList, error) {
	var pl PeerList

	resp, err := http.Get(url)
	if err != nil {
		log.Warnf("Failed getting file: %s", err)
		return nil, err
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Warnf("Failed reading file: %s", err)
		return nil, err
	}

	err = json.Unmarshal(body, &pl)
	if err != nil {
		log.Warnf("Failed parsing json: %s", err)
		return nil, err
	}

	if hl.Cap() < len(pl) {
		log.Warnf("PeerList larger than capacity", err)
		return nil, errNoEnoughCapacity
	}

	return pl, nil
}
