package kungfu

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
	"sync"
)

type config struct {
	values map[string]string
}

type configServer struct {
	mu      sync.RWMutex
	updated chan string

	versionList []string
	versions    map[string]*config
}

func NewConfigServer(updated chan string) http.Handler {
	return &configServer{
		updated:  updated,
		versions: make(map[string]*config),
	}
}

func (cs *configServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	peer := req.Header.Get("x-kungfu-peer")
	log.Printf("%s %s %s from %s @ %s", req.Method, req.URL.Path, req.URL.RawQuery, peer, req.RemoteAddr)
	if strings.HasPrefix(req.URL.Path, "/versions/next/") {
		cs.handleGetNextVersion(w, req)
		return
	}
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

type Version struct {
	Version string
	ID      int
}

func (cs *configServer) handleGetNextVersion(w http.ResponseWriter, req *http.Request) {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	l := len(cs.versionList)
	if l == 0 {
		http.NotFound(w, req)
		return
	}
	var n int
	if _, err := fmt.Sscanf(req.URL.Path, "/versions/next/%d", &n); err != nil {
		http.Error(w, "", http.StatusBadRequest)
		return
	}
	n++
	if n < 0 {
		n = 0
	}
	if n >= l {
		http.NotFound(w, req)
		return
	}
	resp := Version{
		Version: cs.versionList[n],
		ID:      n,
	}
	json.NewEncoder(w).Encode(resp)
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
	cfg, ok := cs.versions[version]
	if !ok {
		cfg = &config{
			values: make(map[string]string),
		}
		cs.versions[version] = cfg
	}
	bs, err := ioutil.ReadAll(req.Body)
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
		return
	}
	value := string(bs)
	if v, ok := cfg.values[name]; ok {
		if v != value {
			http.Error(w, "", http.StatusConflict)
		}
		return
	}
	cfg.values[name] = value
	if name == `KUNGFU_CLUSTER_SPEC` {
		cs.versionList = append(cs.versionList, version)
		if cs.updated != nil {
			cs.updated <- version
		}
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
