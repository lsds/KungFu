package plan

import (
	"errors"
	"fmt"
	"net"
	"strconv"
)

// NetAddr is the network address of a Peer
type NetAddr struct {
	IPv4 uint32
	Port uint16
}

func (a NetAddr) ColocatedWith(b NetAddr) bool {
	return a.IPv4 == b.IPv4
}

func (a NetAddr) String() string {
	return net.JoinHostPort(FormatIPv4(a.IPv4), strconv.Itoa(int(a.Port)))
}

func (a NetAddr) SockFile() string {
	return fmt.Sprintf(`/tmp/kungfu-run-%d.sock`, a.Port)
}

func (a NetAddr) WithName(name string) Addr {
	return Addr{
		IPv4: a.IPv4,
		Port: a.Port,
		Name: name,
	}
}

// Addr is the logical address of a named channel
type Addr struct {
	IPv4 uint32
	Port uint16
	Name string
}

func (a Addr) String() string {
	return a.Name + "@" + NetAddr{a.IPv4, a.Port}.String()
}

func (a Addr) NetAddr() NetAddr {
	return NetAddr{
		IPv4: a.IPv4,
		Port: a.Port,
	}
}

func (a Addr) Peer() PeerID {
	return PeerID{
		IPv4: a.IPv4,
		Port: a.Port,
	}
}

func FormatIPv4(ipv4 uint32) string {
	ip := net.IPv4(byte(ipv4>>24), byte(ipv4>>16), byte(ipv4>>8), byte(ipv4))
	return ip.String()
}

var (
	errInvalidIPv4 = errors.New("invalid IPv4")
	errInvalidPort = errors.New("invalid port")
)

func ParseIPv4(host string) (uint32, error) {
	ip := net.ParseIP(host)
	if ip == nil {
		return 0, errInvalidIPv4
	}
	ip = ip.To4()
	if ip == nil {
		return 0, errInvalidIPv4
	}
	return PackIPv4(ip), nil
}

func PackIPv4(ip net.IP) uint32 {
	a := uint32(ip[0]) << 24
	b := uint32(ip[1]) << 16
	c := uint32(ip[2]) << 8
	d := uint32(ip[3])
	return a | b | c | d
}

func MustParseIPv4(host string) uint32 {
	ipv4, err := ParseIPv4(host)
	if err != nil {
		panic(err)
	}
	return ipv4
}
