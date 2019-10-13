package kungfurun

import (
	"errors"
	"fmt"
	"net"
	"strings"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

func InferSelfIPv4(hostname string, nic string) (uint32, error) {
	if len(hostname) > 0 {
		return plan.ParseIPv4(hostname)
	}
	if len(nic) > 0 {
		return inferIPv4(nic)
	}
	return plan.MustParseIPv4(`127.0.0.1`), nil
}

var errNoIPv4Found = errors.New("no ipv4 found")

func inferIPv4(nic string) (uint32, error) {
	ifaces, err := net.Interfaces()
	if err != nil {
		return 0, err
	}
	for _, i := range ifaces {
		if i.Name != nic {
			continue
		}
		addrs, err := i.Addrs()
		if err != nil {
			continue
		}
		for _, addr := range addrs {
			var ip net.IP
			switch v := addr.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}
			if ip != nil {
				ip = ip.To4()
			}
			if ip != nil {
				return plan.PackIPv4(ip), nil
			}
		}
	}
	return 0, errNoIPv4Found
}

var (
	errNicNotFound         = errors.New("nic not found")
	errNicHasNoIPv4Network = errors.New("nic has no ipv4 network")
)

func getIPv4Net(nic string) (*net.IPNet, error) {
	ifaces, err := net.Interfaces()
	if err != nil {
		return nil, err
	}
	for _, i := range ifaces {
		if i.Name == nic {
			addrs, err := i.Addrs()
			if err != nil {
				return nil, err
			}
			for _, addr := range addrs {
				if v, ok := addr.(*net.IPNet); ok {
					return v, nil
				}
			}
			return nil, errNicHasNoIPv4Network
		}
	}
	return nil, errNicNotFound
}

type HostSpec struct {
	Hostname   string
	Slots      int
	PublicAddr string
}

func parseHostSpec(config string) (*HostSpec, error) {
	var hostname, pubAddr string
	var slots int
	parts := strings.Split(config, ":")
	if len(parts) < 1 {
		return nil, plan.ErrInvalidHostSpec
	}
	hostname = parts[0]
	if len(parts) > 1 {
		if _, err := fmt.Sscanf(parts[1], "%d", &slots); err != nil {
			return nil, plan.ErrInvalidHostSpec
		}
	}
	if len(parts) > 2 {
		pubAddr = parts[2]
	} else {
		pubAddr = hostname
	}
	if len(parts) > 3 {
		return nil, plan.ErrInvalidHostSpec
	}
	return &HostSpec{Hostname: hostname, Slots: slots, PublicAddr: pubAddr}, nil
}

func parseHostList(config string) ([]HostSpec, error) {
	var hl []HostSpec
	for _, h := range strings.Split(config, ",") {
		spec, err := parseHostSpec(h)
		if err != nil {
			return nil, err
		}
		hl = append(hl, *spec)
	}
	return hl, nil
}

func lookupIPv4(host string) []net.IP {
	ips, err := net.LookupIP(host)
	if err != nil {
		return nil
	}
	var ipv4s []net.IP
	for _, ip := range ips {
		if ip := ip.To4(); ip != nil {
			ipv4s = append(ipv4s, ip)
		}
	}
	return ipv4s
}

var errFailedToResoveIPv4 = errors.New("failed to resolve ipv4")

func resolveIPv4(domainOrIPv4 string, ipv4net *net.IPNet) (uint32, error) {
	if ipv4, err := plan.ParseIPv4(domainOrIPv4); err == nil {
		return ipv4, nil
	}
	var ipv4s []uint32
	for _, ip := range lookupIPv4(domainOrIPv4) {
		if ipv4net.Contains(ip) {
			ipv4s = append(ipv4s, plan.PackIPv4(ip))
		}
	}
	if n := len(ipv4s); n != 1 {
		if n > 1 {
			log.Errorf("multiple ipv4 for %s detected in %s", domainOrIPv4, *ipv4net)
		} else {
			log.Errorf("no ipv4 for %s detected in %s", domainOrIPv4, *ipv4net)
		}
		return 0, errFailedToResoveIPv4
	}
	log.Errorf("%s resolved to %s", domainOrIPv4, plan.FormatIPv4(ipv4s[0]))
	return ipv4s[0], nil
}

func resolveHostList(hl []HostSpec, nic string) (plan.HostList, error) {
	ipv4net, err := getIPv4Net(nic)
	if err != nil {
		return nil, err
	}
	var hostlist plan.HostList
	for _, h := range hl {
		ipv4, err := resolveIPv4(h.Hostname, ipv4net)
		if err != nil {
			return nil, err
		}
		hostlist = append(hostlist, plan.HostSpec{IPv4: ipv4, Slots: h.Slots, PublicAddr: h.PublicAddr})
	}
	return hostlist, nil
}

func ParseHostList(config string) ([]HostSpec, error) {
	return parseHostList(config)
}

func ResolveHostList(config string, nic string) (plan.HostList, error) {
	hl, err := parseHostList(config)
	if err != nil {
		return nil, err
	}
	var needResolve bool
	for _, h := range hl {
		if !isIPv4(h.Hostname) {
			needResolve = true
			break
		}
	}
	if needResolve {
		return resolveHostList(hl, nic)
	}
	return plan.ParseHostList(config)
}

func isIPv4(s string) bool {
	_, err := plan.ParseIPv4(s)
	return err == nil
}
