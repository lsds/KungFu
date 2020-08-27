package hostfile

import (
	"errors"
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"

	"github.com/lsds/KungFu/srcs/go/plan"
)

// ParseFile parses -hostfile: https://www.open-mpi.org/doc/current/man1/mpirun.1.php
func ParseFile(filename string) (plan.HostList, error) {
	bs, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return Parse(string(bs))
}

func Parse(text string) (plan.HostList, error) {
	var hl plan.HostList
	for _, line := range strings.Split(text, "\n") {
		line := trimComment(line)
		line = strings.TrimSpace(line)
		if len(line) <= 0 || strings.HasPrefix(line, "#") {
			continue
		}
		h, err := parseLine(line)
		if err != nil {
			return nil, err
		}
		hl = append(hl, *h)
	}
	return hl, nil
}

var errInvalidHostfile = errors.New("invalid hostfile")

func parseLine(line string) (*plan.HostSpec, error) {
	parts := strings.Split(line, " ")
	if len(parts) < 1 {
		return nil, errInvalidHostfile
	}
	ipv4, err := plan.ParseIPv4(parts[0])
	if err != nil {
		return nil, fmt.Errorf("%v: %q", err, parts[0])
	}
	slots := 1
	pubAddr := plan.FormatIPv4(ipv4)
	for _, kv := range parts[1:] {
		kvs := strings.Split(kv, "=")
		if len(kvs) != 2 {
			return nil, errInvalidHostfile
		}
		k, v := kvs[0], kvs[1]
		switch k {
		case `slots`:
			n, err := strconv.Atoi(v)
			if err != nil {
				return nil, errInvalidHostfile
			}
			slots = n
		case `public_addr`:
			pubAddr = v
		default:
			return nil, errInvalidHostfile
		}
	}
	return &plan.HostSpec{
		IPv4:       ipv4,
		Slots:      slots,
		PublicAddr: pubAddr,
	}, nil
}

func trimComment(line string) string {
	parts := strings.SplitN(line, "#", 2)
	return parts[0]
}
