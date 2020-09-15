package configserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type ConfigServer struct {
	sync.RWMutex
	cancel  context.CancelFunc
	Path    string
	mux     http.ServeMux
	cluster *plan.Cluster
	version int
}

func New(cancel context.CancelFunc, initCluster *plan.Cluster, path string) http.Handler {
	s := &ConfigServer{
		Path:    path,
		cluster: initCluster,
		cancel:  cancel,
	}
	s.mux.HandleFunc(s.Path, http.HandlerFunc(s.handleConfig))
	s.mux.HandleFunc(`/stop`, http.HandlerFunc(s.stop))
	return s
}

func (s *ConfigServer) stop(w http.ResponseWriter, req *http.Request) {
	s.cancel()
}

func (s *ConfigServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	s.mux.ServeHTTP(w, req)
}

func (s *ConfigServer) handleConfig(w http.ResponseWriter, req *http.Request) {
	switch req.Method {
	case http.MethodGet:
		s.getConfig(w, req)
	case http.MethodPost:
		s.resetConfig(w, req)
	case http.MethodPut:
		s.putConfig(w, req)
	case http.MethodDelete:
		s.deleteConfig(w, req)
	default:
		http.Error(w, "unsupported method", http.StatusMethodNotAllowed)
	}
}

func (s *ConfigServer) getConfig(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	s.RLock()
	defer s.RUnlock()
	if s.cluster == nil {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintf(w, "No Config Found.\n")
		return
	}
	e := json.NewEncoder(w)
	e.SetIndent("", "    ")
	if err := e.Encode(s.cluster); err != nil {
		log.Errorf("failed to encode JSON: %v", err)
	}
}

func (s *ConfigServer) putConfig(w http.ResponseWriter, req *http.Request) {
	var cluster plan.Cluster
	if err := utils.ReadJSON(req.Body, &cluster); err != nil {
		log.Errorf("failed to decode JSON: %v", err)
		return
	}
	if err := cluster.Validate(); err != nil {
		log.Errorf("invalid cluster config: %v", err)
		return
	}
	s.Lock()
	defer s.Unlock()
	if s.cluster == nil {
		log.Infof("init first config to %d peers: %s", len(cluster.Workers), cluster)
		s.version = 1
		s.cluster = &cluster
	} else if len(s.cluster.Workers) > 0 {
		s.version++
		s.cluster = &cluster
		log.Infof("updated to %d peers: %s", len(cluster.Workers), cluster.Workers)
	} else {
		log.Infof("config was cleared, update rejected")
		w.WriteHeader(http.StatusForbidden)
	}
}

func (s *ConfigServer) resetConfig(w http.ResponseWriter, req *http.Request) {
	s.Lock()
	defer s.Unlock()
	s.cluster = nil
	log.Infof("OK: reset config")
}

func (s *ConfigServer) deleteConfig(w http.ResponseWriter, req *http.Request) {
	s.Lock()
	defer s.Unlock()
	s.cluster = nil
	log.Warnf("config deleted!")
}
