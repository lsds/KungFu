package plan

import (
	"bytes"
	"encoding/binary"
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
	return fmt.Sprintf("[%d/%d]{%s}{%s}", len(c.Workers), len(c.Runners), c.Workers, c.Runners)
}
