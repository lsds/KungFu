package plan

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
)

type Cluster struct {
	Runners PeerList
	Workers PeerList
}

func (c *Cluster) Eq(d Cluster) bool {
	return c.Runners.Eq(d.Runners) && c.Workers.Eq(d.Workers)
}

func (c Cluster) Bytes() []byte {
	b := &bytes.Buffer{}
	for _, h := range c.Runners {
		binary.Write(b, binary.LittleEndian, &h)
	}
	for _, p := range c.Workers {
		binary.Write(b, binary.LittleEndian, &p)
	}
	return b.Bytes()
}

func (c Cluster) DebugString() string {
	return fmt.Sprintf("[%d@%d]{%s}@{%s}", len(c.Workers), len(c.Runners), c.Workers, c.Runners)
}

var (
	errDuplicatedPort   = errors.New("duplicated port")
	errDuplicatedRunner = errors.New("duplicated runner")
	errMissingRunner    = errors.New("missing runner")
)

func (c Cluster) Validate() error {
	h := make(map[uint32]int)
	p := make(map[PeerID]int)
	for _, r := range c.Runners {
		if p[r] != 0 {
			return errDuplicatedPort
		}
		p[r]++
		if h[r.IPv4] != 0 {
			return errDuplicatedRunner
		}
		h[r.IPv4]++
	}
	for _, w := range c.Workers {
		if p[w] != 0 {
			return errDuplicatedPort
		}
		p[w]++
		if h[w.IPv4] == 0 {
			return errMissingRunner
		}
	}
	return nil
}
