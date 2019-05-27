package rchannel

import (
	//"fmt"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
)



type ModelStore struct {
	modelStore      map[string]*kb.Buffer
	modelStoreMutex sync.Mutex

}

func NewModelStore() *ModelStore {
    s := &ModelStore{
			modelStore : make(map[string]*kb.Buffer),
	}
	return s
}



func (store ModelStore) UpdateModelStore(varname string, varbuf *kb.Buffer) error {
	//fmt.Println("Locking when updating the model store")
	store.modelStoreMutex.Lock()
	defer store.modelStoreMutex.Unlock()

	//fmt.Printf("Updating model store for variable: %s\n", varname)
	//fmt.Printf("Contents of the variables: %+v\n", varbuf.Data[:20])

	_, ok := store.modelStore[varname]
	if !ok {
		log.Warnf("%s has no entry in the store. Initializing storage for this variable.", varname)
		newBuf := kb.NewBuffer(varbuf.Count, varbuf.Type)
		newBuf.CopyFrom(varbuf)
		store.modelStore[varname] = newBuf
	} else {
		//log.Infof("%s copying from buffer to model store", varname)
		store.modelStore[varname].CopyFrom(varbuf)
	}

	return nil
}