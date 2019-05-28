package rchannel

import (
	//"fmt"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
)



type ModelStore struct {
	modelStore      *kb.Buffer
	modelStoreMutex sync.Mutex

}

func NewModelStore() *ModelStore {
    s := &ModelStore{
			modelStore : nil,
	}
	return s
}



func (store *ModelStore) UpdateModelStore(updateName string, model *kb.Buffer) error {
	store.modelStoreMutex.Lock()
	defer store.modelStoreMutex.Unlock()

	//fmt.Printf("Updating model store: %+v\n", model.Data[:20])

	if store.modelStore == nil {
		log.Warnf("%s has no entry in the store. Initializing storage for this variable.", updateName)
		newBuf := kb.NewBuffer(model.Count, model.Type)
		newBuf.CopyFrom(model)
		store.modelStore = newBuf
	} else {
		store.modelStore.CopyFrom(model)
	}

	return nil
}