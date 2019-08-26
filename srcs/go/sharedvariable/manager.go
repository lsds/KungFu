package sharedvariable

import (
	"errors"
	"log"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

var (
	errConflict = errors.New("conflict")
	errNotFound = errors.New("not found")
)

type SharedVariableManager struct {
	sync.RWMutex

	vars map[string]*SharedVariable
}

func NewSharedVariableManager() *SharedVariableManager {
	return &SharedVariableManager{
		vars: make(map[string]*SharedVariable),
	}
}

func (m *SharedVariableManager) get(name string) (*SharedVariable, error) {
	m.RLock()
	defer m.RUnlock()
	v, ok := m.vars[name]
	if !ok {
		return nil, errNotFound
	}
	return v, nil
}

func (m *SharedVariableManager) Create(name string, count int, dtype kb.KungFu_Datatype) (*SharedVariable, error) {
	log.Printf("SharedVariableManager::Create(%s, %d, %d)", name, count, dtype)
	if v, err := m.get(name); err == nil {
		if v.data.Count != count || v.data.Type != dtype {
			return nil, errConflict
		}
		return v, nil
	}
	m.Lock()
	defer m.Unlock()
	v := NewSharedVariable(count, dtype)
	m.vars[name] = v
	return v, nil
}

func (m *SharedVariableManager) Get(name string, buf *kb.Buffer) error {
	log.Printf("SharedVariableManager::Get(%s, %d, %s)", name, buf.Count, buf.Type)
	v, err := m.get(name)
	if err != nil {
		return err
	}
	return v.Get(buf)
}

func (m *SharedVariableManager) Put(name string, buf *kb.Buffer) error {
	log.Printf("SharedVariableManager::Put(%s, %d, %s)", name, buf.Count, buf.Type)
	v, err := m.get(name)
	if err != nil {
		return err
	}
	return v.Put(buf)
}

func (m *SharedVariableManager) Add(name string, buf *kb.Buffer, output *kb.Buffer) error {
	log.Printf("SharedVariableManager::Add(%s, %d, %s)", name, buf.Count, buf.Type)
	v, err := m.get(name)
	if err != nil {
		return err
	}
	return v.Add(buf, output)
}

// f must be readonly
func (m *SharedVariableManager) do(name string, f func(*kb.Buffer)) error {
	// log.Printf("SharedVariableManager::Get(%s, %d, %d)", name, buf.Count, buf.Type)
	v, err := m.get(name)
	if err != nil {
		return err
	}
	return v.do(f)
}
