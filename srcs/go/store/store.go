package store

import (
	"errors"
	"sync"
)

var (
	errConflict = errors.New("conflict")
	errNotFound = errors.New("not found")
)

// Store is a simple Key-Value store
type Store struct {
	sync.RWMutex

	data map[string]*Blob
}

func NewStore() *Store {
	return &Store{
		data: make(map[string]*Blob),
	}
}

func (s *Store) Create(name string, size int) (*Blob, error) {
	s.Lock()
	defer s.Unlock()
	if _, ok := s.data[name]; ok {
		return nil, errConflict
	}
	blob := NewBlob(size)
	s.data[name] = blob
	return blob, nil
}

func (s *Store) Get(name string) (*Blob, error) {
	s.RLock()
	defer s.RUnlock()
	blob, ok := s.data[name]
	if !ok {
		return nil, errNotFound
	}
	return blob, nil
}

func (s *Store) GetOrCreate(name string, size int) (*Blob, error) {
	s.Lock()
	defer s.Unlock()
	if blob, ok := s.data[name]; ok {
		if len(blob.Data) == size {
			return blob, nil
		}
		return nil, errConflict
	}
	blob := NewBlob(size)
	s.data[name] = blob
	return blob, nil
}
