package plan

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

// FIXME: make members private, public is required by JSON encoding for now

var errInvalidHostSpec = errors.New("Invalid HostSpec")

type HostSpec struct {
	Hostname   string
	Slots      int
	PublicAddr string
}

func DefaultHostSpec() HostSpec {
	return HostSpec{
		Hostname:   `127.0.0.1`,
		Slots:      runtime.NumCPU(),
		PublicAddr: `127.0.0.1`,
	}
}

func (h HostSpec) String() string {
	return fmt.Sprintf("%s:%d:%s", h.Hostname, h.Slots, h.PublicAddr)
}

func parseHostSpec(spec string) (*HostSpec, error) {
	parts := strings.Split(spec, ":")
	switch len(parts) {
	case 1:
		return &HostSpec{Hostname: parts[0], Slots: 1, PublicAddr: parts[0]}, nil
	case 2:
		slots, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, errInvalidHostSpec
		}
		return &HostSpec{Hostname: parts[0], Slots: slots, PublicAddr: parts[0]}, nil
	case 3:
		slots, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, errInvalidHostSpec
		}
		return &HostSpec{Hostname: parts[0], Slots: slots, PublicAddr: parts[2]}, nil
	}
	return nil, errInvalidHostSpec
}

type HostList []HostSpec

func (hl HostList) String() string {
	var ss []string
	for _, h := range hl {
		ss = append(ss, h.String())
	}
	return strings.Join(ss, ",")
}

func ParseHostList(hostlist string) (HostList, error) {
	var hostSpecs HostList
	for _, h := range strings.Split(hostlist, ",") {
		spec, err := parseHostSpec(h)
		if err != nil {
			return nil, err
		}
		hostSpecs = append(hostSpecs, *spec)
	}
	return hostSpecs, nil
}

func TotalCap(hostSpecs []HostSpec) int {
	var cap int
	for _, h := range hostSpecs {
		cap += h.Slots
	}
	return cap
}

func GetHostListFromEnv() (HostList, error) {
	val := os.Getenv(kb.HostListEnvKey)
	return ParseHostList(val)
}
