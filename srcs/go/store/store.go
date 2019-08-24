package store

import (
	"errors"
	"sync"
)

var (
	errReadConflict  = errors.New("read conflict")
	errWriteConflict = errors.New("write conflict")
	errNotFound      = errors.New("not found")
	errSizeNotMatch  = errors.New("size not match")
)

type Blob struct {
	Data []byte
}

func NewBlob(n int) *Blob {
	return &Blob{Data: make([]byte, n)}
}

func (b *Blob) copyFrom(c *Blob) error {
	if len(b.Data) != len(c.Data) {
		return errSizeNotMatch
	}
	copy(b.Data, c.Data)
	return nil
}

// Store is a simple Key-Value store
type Store struct {
	sync.RWMutex

	data map[string]*Blob
}

func newStore() *Store {
	return &Store{
		data: make(map[string]*Blob),
	}
}

func (s *Store) Create(name string, buf *Blob) error {
	s.Lock()
	defer s.Unlock()
	if _, ok := s.data[name]; ok {
		return errWriteConflict
	}
	value := NewBlob(len(buf.Data))
	value.copyFrom(buf)
	s.data[name] = value
	return nil
}

// Get retrives the data with given name, if blob is not nil,
// the length of blob.Data is used to validate the stored data
func (s *Store) Get(name string, blob **Blob) error {
	s.RLock()
	defer s.RUnlock()
	value, ok := s.data[name]
	if !ok {
		return errNotFound
	}
	if *blob == nil {
		*blob = NewBlob(len(value.Data))
	}
	if err := (*blob).copyFrom(value); err != nil {
		return errReadConflict
	}
	return nil
}
