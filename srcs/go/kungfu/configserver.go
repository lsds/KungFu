package kungfu

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"strings"

	"github.com/lsds/KungFu/srcs/go/store"
)

type configServer struct {
	updated chan string

	versionList []string
	store       *store.VersionedStore
}

func NewConfigServer(updated chan string) http.Handler {
	return &configServer{
		updated: updated,
		store:   store.NewVersionedStore(0),
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

func (cs *configServer) handleGetNextVersion(w http.ResponseWriter, req *http.Request) {
	parts := strings.Split(req.URL.Path, "/")
	if len(parts) != 4 {
		http.Error(w, "", http.StatusBadRequest)
		return
	}
	nextVersion := cs.store.GetNextVersion(parts[3])
	w.Write([]byte(nextVersion))
}

func (cs *configServer) handleReadConfig(w http.ResponseWriter, req *http.Request) {
	version := req.FormValue("version")
	name := req.FormValue("name")
	var blob *store.Blob
	if err := cs.store.Get(version, name, &blob); err != nil {
		http.NotFound(w, req)
		return
	}
	w.Write(blob.Data)
}

func (cs *configServer) handleWriteConfig(w http.ResponseWriter, req *http.Request) {
	version := req.FormValue("version")
	name := req.FormValue("name")
	bs, err := ioutil.ReadAll(req.Body)
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
		return
	}
	{
		var blob *store.Blob
		if err := cs.store.Get(version, name, &blob); err == nil {
			if string(bs) != string(blob.Data) {
				http.Error(w, "", http.StatusConflict)
			}
			return
		}
	}
	blob := &store.Blob{Data: bs}
	if err := cs.store.Create(version, name, blob); err != nil {
		http.Error(w, "", http.StatusConflict)
		return
	}
	if name == `KUNGFU_CLUSTER_SPEC` {
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
