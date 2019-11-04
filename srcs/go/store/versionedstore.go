package store

import (
	"errors"
	"sync"
)

type VersionedStore struct {
	sync.RWMutex

	versions map[string]*Store

	windowSize int
	window     []string
}

var errInvalidWindowSize = errors.New("invalid window size")

func NewVersionedStore(windowSize int) *VersionedStore {
	return &VersionedStore{
		versions:   make(map[string]*Store),
		windowSize: windowSize,
	}
}

func (s *VersionedStore) getVersion(version string) (*Store, error) {
	s.RLock()
	defer s.RUnlock()
	store, ok := s.versions[version]
	if !ok {
		return nil, errNotFound
	}
	return store, nil
}

func (s *VersionedStore) gc() {
	if s.windowSize <= 0 {
		return
	}
	for len(s.window) > s.windowSize {
		old := s.window[0]
		s.window = s.window[1:]
		delete(s.versions, old)
	}
}

func (s *VersionedStore) getOrCreateVersion(version string) *Store {
	s.Lock()
	defer s.Unlock()
	store, ok := s.versions[version]
	if !ok {
		store = NewStore()
		s.versions[version] = store
		s.window = append(s.window, version)
		s.gc()
	}
	return store
}

func (s *VersionedStore) Create(version, name string, size int) (*Blob, error) {
	store := s.getOrCreateVersion(version)
	return store.Create(name, size)
}

func (s *VersionedStore) Get(version, name string) (*Blob, error) {
	store, err := s.getVersion(version)
	if err != nil {
		return nil, err
	}
	return store.Get(name)
}

func (s *VersionedStore) GetOrCreate(version, name string, size int) (*Blob, error) {
	store := s.getOrCreateVersion(version)
	return store.GetOrCreate(name, size)
}

func (s *VersionedStore) GetNextVersion(prev string) string {
	s.RLock()
	defer s.RUnlock()
	n := len(s.window)
	for i := n - 1; i > 0; i-- {
		if prev == s.window[i-1] {
			return s.window[i]
		}
	}
	if n > 0 {
		if prev == s.window[n-1] {
			return prev
		}
		return s.window[0]
	}
	return prev
}
