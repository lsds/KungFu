package rchannel

import (
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
)

type ModelStore struct {
	sync.Mutex

	data map[string]*kb.Buffer
}

func newModelStore() *ModelStore {
	return &ModelStore{
		data: make(map[string]*kb.Buffer),
	}
}

func (s *ModelStore) Update(name string, buf *kb.Buffer) {
	s.Lock()
	defer s.Unlock()
	if _, ok := s.data[name]; !ok {
		log.Warnf("%s has no entry in the store, init as %s[%d].", name, buf.Type /* TODO: show dtype name*/, buf.Count)
		s.data[name] = kb.NewBuffer(buf.Count, buf.Type)
		// TODO: support GC
	} else {
		// TODO: check shape here
	}
	s.data[name].CopyFrom(buf)
}
