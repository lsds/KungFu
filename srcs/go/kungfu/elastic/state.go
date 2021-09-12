package elastic

import (
	"bytes"
	"fmt"
)

//
type State struct {
	Rank      int      `json:"rank"`
	Size      int      `json:"size"`
	Progress  uint64   `json:"progress"`
	Filenames []string `json:"filenames"`
}

func (s State) String() string {
	b := &bytes.Buffer{}
	fmt.Fprintf(b, "<ElasticState(%d/%d@%d)>", s.Rank, s.Size, s.Progress)
	return b.String()
}
