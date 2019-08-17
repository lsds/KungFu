package store

import (
	"errors"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

var (
	errReadConflict  = errors.New("read conflict")
	errWriteConflict = errors.New("write conflict")
	errNotFound      = errors.New("not found")
)

// Store is a simple Key-Value stor
type Store struct {
	sync.Mutex

	data map[string]*kb.Buffer
}

func newStore() *Store {
	return &Store{
		data: make(map[string]*kb.Buffer),
	}
}

func (s *Store) Create(name string, buf *kb.Buffer) error {
	s.Lock()
	defer s.Unlock()
	if _, ok := s.data[name]; ok {
		return errWriteConflict
	}
	value := kb.NewBuffer(buf.Count, buf.Type)
	value.CopyFrom(buf)
	s.data[name] = value
	return nil
}

// Get retrives the data with given name, if buf is not nil,
// the metadata of buf is used to validate the stored data
func (s *Store) Get(name string, buf *kb.Buffer) error {
	s.Lock()
	defer s.Unlock()
	value, ok := s.data[name]
	if !ok {
		return errNotFound
	}
	if buf == nil {
		buf = kb.NewBuffer(value.Count, value.Type)
	}
	if err := buf.MaybeCopyFrom(value); err != nil {
		return errReadConflict
	}
	return nil
}
