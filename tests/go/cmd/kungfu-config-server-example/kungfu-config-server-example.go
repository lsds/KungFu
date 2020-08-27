// deprecated, should use srcs/go/cmd/kungfu-config-server/kungfu-config-server.go
package main

import (
	"context"
	"encoding/json"
	"expvar"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	port     = flag.Int("port", 9100, "")
	initFile = flag.String("init", "", "")
	ttl      = flag.Duration("ttl", 0, "time to live")
)

type configServer struct {
	sync.RWMutex
	cancel context.CancelFunc

	cluster *plan.Cluster
	version int
}

func (s *configServer) index(w http.ResponseWriter, req *http.Request) {
	index := `<!doctype html><html><body><ul>
		<li><a target="_blank" href="/get">/get</a>: get current config</li>
		<li><a target="_blank">/put</a>: put new config</li>
		<li><a target="_blank" href="/clear">/clear</a>: set config to empty list, and reject all later configs. Should cause all workers to stop.</li>
		<li><a target="_blank" href="/reset">/reset</a>: set config to nil, and start accepting new configs. Workers should ignore nil config and keep using existing config.</li>
		<li><a target="_blank" href="/debug/vars">/debug/vars</a></li>
	<ul></body></html>`
	w.Write([]byte(index))
}

func (s *configServer) getConfig(w http.ResponseWriter, req *http.Request) {
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

func (s *configServer) putConfig(w http.ResponseWriter, req *http.Request) {
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
		log.Infof("config is cleared, update rejected")
		w.WriteHeader(http.StatusForbidden)
	}
}

func (s *configServer) clearConfig(w http.ResponseWriter, req *http.Request) {
	s.Lock()
	defer s.Unlock()
	s.cluster = &plan.Cluster{}
	log.Infof("OK: clear config")
}

func (s *configServer) resetConfig(w http.ResponseWriter, req *http.Request) {
	s.Lock()
	defer s.Unlock()
	s.cluster = nil
	log.Infof("OK: reset config")
}

func (s *configServer) stop(w http.ResponseWriter, req *http.Request) {
	s.cancel()
}

func (s *configServer) removeWorker(w http.ResponseWriter, req *http.Request) {
	s.Lock()
	defer s.Unlock()
	var cluster = s.cluster.Clone()
	var peer plan.PeerID
	if err := utils.ReadJSON(req.Body, &peer); err != nil {
		log.Errorf("failed to decode JSON: %v", err)
		return
	}
	log.Infof("%s wants to remove %s", req.UserAgent(), peer)
	numWorkersSameIP := 0
	for i, worker := range cluster.Workers {
		if peer == worker {
			cluster.Workers[i] = cluster.Workers[len(cluster.Workers)-1]
			cluster.Workers = cluster.Workers[:len(cluster.Workers)-1]
		} else {
			if peer.IPv4 == worker.IPv4 {
				numWorkersSameIP = numWorkersSameIP + 1
			}
		}
	}
	if numWorkersSameIP == 0 {
		for i, runner := range s.cluster.Runners {
			if peer.IPv4 == runner.IPv4 {
				cluster.Runners[i] = cluster.Runners[len(cluster.Runners)-1]
				cluster.Runners = cluster.Runners[:len(cluster.Runners)-1]
			}
		}
	}
	if err := cluster.Validate(); err != nil {
		log.Errorf("invalid cluster config: %v", err)
		return
	}
	s.cluster = &cluster
	log.Infof("after remove worker: %s requested by %s", cluster, req.UserAgent())
}

func (s *configServer) addWorker(w http.ResponseWriter, req *http.Request) {
	s.Lock()
	defer s.Unlock()
	var cluster = s.cluster.Clone()
	var peer plan.PeerID
	if err := utils.ReadJSON(req.Body, &peer); err != nil {
		log.Errorf("failed to decode JSON: %v", err)
		return
	}
	cluster.Workers = append(cluster.Workers, peer)
	numRunners := 0
	for _, runner := range s.cluster.Runners {
		if peer.IPv4 == runner.IPv4 {
			numRunners = numRunners + 1
		}
	}
	if numRunners == 0 {
		newRunner := peer
		newRunner.Port = 38080
		cluster.Runners = append(cluster.Runners, newRunner)
	}
	if err := cluster.Validate(); err != nil {
		log.Errorf("invalid cluster config: %v", err)
		return
	}
	s.cluster = &cluster
	log.Infof("addWorker worker: %s", cluster)
}

func main() {
	t0 := time.Now()
	flag.Parse()
	log.Errorf("deprecated! Please use kungfu-config-server")
	ctx, cancel := context.WithCancel(context.Background())
	h := &configServer{
		cancel: cancel,
	}
	if len(*initFile) > 0 {
		f, err := os.Open(*initFile)
		if err != nil {
			utils.ExitErr(err)
		}
		defer f.Close()
		var cluster plan.Cluster
		if err := utils.ReadJSON(f, &cluster); err != nil {
			utils.ExitErr(err)
		}
		h.cluster = &cluster
	}

	log.Infof("http://127.0.0.1:%d/", *port)

	mux := &http.ServeMux{}
	mux.HandleFunc("/", http.HandlerFunc(h.index))
	mux.HandleFunc("/get", http.HandlerFunc(h.getConfig))
	mux.HandleFunc("/put", http.HandlerFunc(h.putConfig))
	mux.HandleFunc("/reset", http.HandlerFunc(h.resetConfig))
	mux.HandleFunc("/clear", http.HandlerFunc(h.clearConfig))
	mux.HandleFunc("/stop", http.HandlerFunc(h.stop))
	mux.HandleFunc("/removeworker", http.HandlerFunc(h.removeWorker))
	mux.HandleFunc("/addworker", http.HandlerFunc(h.addWorker))
	mux.Handle("/debug/vars", expvar.Handler())
	if *ttl > 0 {
		ctx, cancel = context.WithTimeout(ctx, *ttl)
		defer cancel()
	}
	srv := &http.Server{
		Addr:    net.JoinHostPort("", strconv.Itoa(*port)),
		Handler: logRequest(mux),
	}
	srv.SetKeepAlivesEnabled(false)
	go func() {
		if err := srv.ListenAndServe(); err != nil {
			utils.ExitErr(err)
		}
	}()
	<-ctx.Done()
	srv.Close()
	log.Infof("%s stopped after %s", utils.ProgName(), time.Since(t0))
}

func logRequest(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		log.Debugf("%s %s from %s, UA: %s", req.Method, req.URL.Path, req.RemoteAddr, req.UserAgent())
		h.ServeHTTP(w, req)
	})
}
