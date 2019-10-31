package rchannel

import (
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
)

type LocalStore struct {
	sync.RWMutex

	data map[string]*kb.Vector
}

func newLocalStore() *LocalStore {
	return &LocalStore{
		data: make(map[string]*kb.Vector),
	}
}

func (s *LocalStore) Emplace(name string, buf *kb.Vector) {
	s.Lock()
	defer s.Unlock()
	if _, ok := s.data[name]; !ok {
		log.Debugf("%s has no entry in the store, init as %s[%d].", name, buf.Type, buf.Count)
		s.data[name] = kb.NewVector(buf.Count, buf.Type)
		// TODO: support GC
	} else {
		// TODO: check shape here
	}
	s.data[name].CopyFrom(buf)
}
