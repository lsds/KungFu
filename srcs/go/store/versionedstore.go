package store

import (
	"errors"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type VersionedStore struct {
	sync.Mutex

	versions map[string]*Store

	windowSize int
	window     []string
}

var errInvalidWindowSize = errors.New("invalid window size")

func NewVersionedStore(windowSize int) *VersionedStore {
	if windowSize < 1 {
		utils.ExitErr(errInvalidWindowSize)
	}
	return &VersionedStore{
		versions:   make(map[string]*Store),
		windowSize: windowSize,
	}
}

func (s *VersionedStore) getVersion(version string) (*Store, error) {
	s.Lock()
	defer s.Unlock()
	store, ok := s.versions[version]
	if !ok {
		return nil, errNotFound
	}
	return store, nil
}

func (s *VersionedStore) getOrCreateVersion(version string) *Store {
	s.Lock()
	defer s.Unlock()
	store, ok := s.versions[version]
	if !ok {
		store = newStore()
		s.versions[version] = store
		s.window = append(s.window, version)
		for len(s.window) > s.windowSize {
			old := s.window[0]
			s.window = s.window[1:]
			delete(s.versions, old)
		}
	}
	return store
}

func (s *VersionedStore) Commit(version, name string, buf *kb.Buffer) error {
	store := s.getOrCreateVersion(version)
	// TODO: gc old version
	return store.Create(name, buf)
}

// Checkout retrives the data with given version and name, if buf is not nil,
// the metadata of buf is used to validate the stored data
func (s *VersionedStore) Checkout(version, name string, buf *kb.Buffer) error {
	store, err := s.getVersion(version)
	if err != nil {
		return err
	}
	return store.Get(name, buf)
}
