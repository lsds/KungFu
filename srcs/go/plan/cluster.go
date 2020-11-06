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

func (c Cluster) Clone() Cluster {
	return Cluster{
		Runners: c.Runners.Clone(),
		Workers: c.Workers.Clone(),
	}
}

var errNoRunnerInCluster = errors.New("no runner in cluster")

// append one worker to the runner which has the minimal number of workers
func (c *Cluster) growOne() error {
	if len(c.Runners) == 0 {
		return errNoRunnerInCluster
	}
	usedSlots := make(map[uint32]int)
	for _, r := range c.Runners {
		usedSlots[r.IPv4] = 0
	}
	for _, w := range c.Workers {
		usedSlots[w.IPv4]++
	}
	ipv4 := c.Runners[0].IPv4
	for _, r := range c.Runners {
		if usedSlots[r.IPv4] < usedSlots[ipv4] {
			ipv4 = r.IPv4
		}
	}
	var port uint16
	for _, w := range c.Workers {
		if w.IPv4 == ipv4 && port <= w.Port {
			port = w.Port + 1
		}
	}
	if port == 0 {
		port = DefaultPortRange.Begin
	}
	newWorker := PeerID{IPv4: ipv4, Port: port}
	c.Workers = append(c.Workers, newWorker)
	return nil
}

func (c Cluster) Resize(newSize int) (*Cluster, error) {
	d := c.Clone()
	if len(d.Workers) > newSize {
		d.Workers = d.Workers[:newSize]
	}
	for i := len(d.Workers); i < newSize; i++ {
		// FIXME: make it more efficient
		if err := d.growOne(); err != nil {
			return nil, err
		}
	}
	return &d, nil
}
