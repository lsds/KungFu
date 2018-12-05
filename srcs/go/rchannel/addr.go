package rchannel

import (
	"net"
)

// Addr is the network address of Endpoint
type NetAddr struct {
	Host string
	Port string
}

func (a NetAddr) String() string {
	return net.JoinHostPort(a.Host, a.Port)
}

func (a NetAddr) WithName(name string) Addr {
	return Addr{
		Host: a.Host,
		Port: a.Port,
		Name: name,
	}
}

// Addr is the address of Endpoint
type Addr struct {
	Host string
	Port string
	Name string
}

func (a Addr) String() string {
	return a.Name + "@" + net.JoinHostPort(a.Host, a.Port)
}
