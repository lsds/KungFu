package kungfu

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type config struct {
	values map[string]string
}

type configServer struct {
	mu       sync.RWMutex
	versions map[string]config
}

func NewConfigServer(hostSpecs []plan.HostSpec, cs *plan.ClusterSpec) http.Handler {
	init := config{
		values: map[string]string{
			kb.HostSpecEnvKey:    toJSON(hostSpecs),
			kb.ClusterSpecEnvKey: toJSON(cs),
		},
	}
	return &configServer{
		versions: map[string]config{
			"": init,
		},
	}
}

func (cs *configServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	log.Printf("%s %s %s", req.Method, req.URL.Path, req.URL.RawQuery)
	switch req.Method {
	case http.MethodGet:
		cs.handleReadConfig(w, req)
		return
	case http.MethodPost:
		cs.handleWriteConfig(w, req)
		return
	default:
		http.Error(w, "", http.StatusMethodNotAllowed)
	}
}

func (cs *configServer) handleReadConfig(w http.ResponseWriter, req *http.Request) {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	version := req.FormValue("version")
	name := req.FormValue("name")
	config, ok := cs.versions[version]
	if !ok {
		http.NotFound(w, req)
		return
	}
	val, ok := config.values[name]
	if !ok {
		http.NotFound(w, req)
		return
	}
	w.Write([]byte(val))
}

func (cs *configServer) handleWriteConfig(w http.ResponseWriter, req *http.Request) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	version := req.FormValue("version")
	name := req.FormValue("name")
	if _, ok := cs.versions[version]; ok {
		http.Error(w, "already exists", http.StatusConflict)
		return
	}
	bs, err := ioutil.ReadAll(req.Body)
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
		return
	}
	cs.versions[version] = config{
		values: map[string]string{
			name: string(bs), // FIXME: check JSON
		},
	}
	w.WriteHeader(http.StatusCreated)
}

func toJSON(i interface{}) string {
	bs, err := json.Marshal(i)
	if err != nil {
		return ""
	}
	return string(bs)
}
