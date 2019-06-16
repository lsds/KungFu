package rchannel

import (
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
)

type LocalStore struct {
	sync.Mutex

	data map[string]*kb.Buffer
}

func newLocalStore() *LocalStore {
	return &LocalStore{
		data: make(map[string]*kb.Buffer),
	}
}

func (s *LocalStore) Emplace(name string, buf *kb.Buffer) {
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
