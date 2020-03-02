package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
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

	cluster *plan.Cluster
	version int
}

func (s *configServer) index(w http.ResponseWriter, req *http.Request) {
	index := `<!doctype html><html><body><ul>
		<li><a target="_blank" href="/get">/get</a>: get current config</li>
		<li><a target="_blank">/put</a>: put new config</li>
		<li><a target="_blank" href="/clear">/clear</a>: set config to empty list, and reject all later configs. Should cause all workers to stop.</li>
		<li><a target="_blank" href="/reset">/reset</a>: set config to nil, and start accepting new configs. Workers should ignore nil config and keep using existing config.</li>
	<ul></body></html>`
	w.Write([]byte(index))
}

func (s *configServer) getConfig(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Printf("%s %s from %s, UA: %s\n", req.Method, req.URL.Path, req.RemoteAddr, req.UserAgent())
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
	if err := readJSON(req.Body, &cluster); err != nil {
		log.Errorf("failed to decode JSON: %v", err)
		return
	}
	s.Lock()
	defer s.Unlock()
	if s.cluster == nil {
		fmt.Printf("init first config to %d peers: %s\n", len(cluster.Workers), cluster)
		s.version = 1
		s.cluster = &cluster
	} else if len(s.cluster.Workers) > 0 {
		s.version++
		s.cluster = &cluster
		fmt.Printf("updated to %d peers: %s\n", len(cluster.Workers), cluster.Workers)
	} else {
		fmt.Printf("config is cleared, update rejected\n")
		w.WriteHeader(http.StatusForbidden)
	}
}

func (s *configServer) clearConfig(w http.ResponseWriter, req *http.Request) {
	s.Lock()
	defer s.Unlock()
	s.cluster = &plan.Cluster{}
	fmt.Printf("OK: clear config\n")
}

func (s *configServer) resetConfig(w http.ResponseWriter, req *http.Request) {
	s.Lock()
	defer s.Unlock()
	s.cluster = nil
	fmt.Printf("OK: reset config\n")
}

func main() {
	t0 := time.Now()
	flag.Parse()
	h := &configServer{}

	if len(*initFile) > 0 {
		f, err := os.Open(*initFile)
		if err != nil {
			utils.ExitErr(err)
		}
		defer f.Close()
		var cluster plan.Cluster
		if err := readJSON(f, &cluster); err != nil {
			utils.ExitErr(err)
		}
		h.cluster = &cluster
	}

	fmt.Printf("http://127.0.0.1:%d/get\n", *port)
	fmt.Printf("http://127.0.0.1:%d/put\n", *port)
	fmt.Printf("http://127.0.0.1:%d/reset\n", *port)

	mux := &http.ServeMux{}
	mux.HandleFunc("/", http.HandlerFunc(h.index))
	mux.HandleFunc("/get", http.HandlerFunc(h.getConfig))
	mux.HandleFunc("/put", http.HandlerFunc(h.putConfig))
	mux.HandleFunc("/reset", http.HandlerFunc(h.resetConfig))
	mux.HandleFunc("/clear", http.HandlerFunc(h.clearConfig))
	ctx := context.Background()
	if *ttl > 0 {
		var cancal context.CancelFunc
		ctx, cancal = context.WithTimeout(ctx, *ttl)
		defer cancal()
	}
	srv := &http.Server{
		Addr:    net.JoinHostPort("", strconv.Itoa(*port)),
		Handler: mux,
	}
	srv.SetKeepAlivesEnabled(false)
	go srv.ListenAndServe()
	<-ctx.Done()
	srv.Close()
	fmt.Printf("Stopped after %s\n", time.Since(t0))
}

func readJSON(r io.Reader, i interface{}) error {
	d := json.NewDecoder(r)
	return d.Decode(&i)
}
