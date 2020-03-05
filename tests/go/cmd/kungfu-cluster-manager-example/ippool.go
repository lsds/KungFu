package main

import "fmt"

type ipPool struct {
	prefix string
	n      int
}

func (p *ipPool) subnet() string {
	return fmt.Sprintf("%s.0/24", p.prefix)
}

func (p *ipPool) get() string {
	x := p.n
	p.n++
	ip := fmt.Sprintf("%s.%d", p.prefix, x)
	fmt.Printf("allocate: %s\n", ip)
	return ip
}
