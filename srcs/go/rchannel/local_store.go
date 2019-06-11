package rchannel

import (
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
)

type ModelStore struct {
	sync.Mutex

	data *kb.Buffer
}

func (s *ModelStore) Update(modelVersionName string, model *kb.Buffer) {
	s.Lock()
	defer s.Unlock()
	if s.data == nil {
		log.Warnf("%s has no entry in the store. Initializing storage for this variable.", modelVersionName)
		s.data = kb.NewBuffer(model.Count, model.Type)
	}
	s.data.CopyFrom(model)
}
