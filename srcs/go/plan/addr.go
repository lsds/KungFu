package plan

import (
	"fmt"
	"net"
	"strconv"
)

// Addr is the network address of a named channel
type NetAddr struct {
	Host string
	Port uint16
}

func (a NetAddr) String() string {
	return net.JoinHostPort(a.Host, strconv.Itoa(int(a.Port)))
}

func (a NetAddr) SockFile() string {
	return fmt.Sprintf(`/tmp/kungfu-prun-%d.sock`, a.Port)
}

func (a NetAddr) WithName(name string) Addr {
	return Addr{
		Host: a.Host,
		Port: a.Port,
		Name: name,
	}
}

func ShmNameFor(from, to NetAddr) string {
	return fmt.Sprintf("kungfu-shm-%s.%d-to-%s.%d", from.Host, from.Port, to.Host, to.Port)
}

// Addr is the logical address of a named channel
type Addr struct {
	Host string
	Port uint16
	Name string
}

func (a Addr) String() string {
	return a.Name + "@" + net.JoinHostPort(a.Host, strconv.Itoa(int(a.Port)))
}

func (a Addr) NetAddr() NetAddr {
	return NetAddr{
		Host: a.Host,
		Port: a.Port,
	}
}
