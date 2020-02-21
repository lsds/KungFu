package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	port     = flag.Int("port", 9100, "")
	initFile = flag.String("init", "", "")
)

type configServer struct {
	sync.RWMutex

	peerList plan.PeerList
}

func (s *configServer) getConfig(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Printf("%s %s from %s %s\n", req.Method, req.URL.Path, req.RemoteAddr, req.UserAgent())
	s.RLock()
	defer s.RUnlock()
	e := json.NewEncoder(w)
	if s.peerList == nil {
		w.WriteHeader(http.StatusNotFound)
	}
	if err := e.Encode(s.peerList); err != nil {
		log.Errorf("failed to encode JSON: %v", err)
	}
}

func (s *configServer) putConfig(w http.ResponseWriter, req *http.Request) {
	var pl plan.PeerList
	if err := readJSON(req.Body, &pl); err != nil {
		log.Errorf("failed to decode JSON: %v", err)
	}
	s.Lock()
	defer s.Unlock()
	if s.peerList == nil {
		fmt.Printf("init first config to %d peers: %s\n", len(pl), pl)
		s.peerList = pl
	} else {
		s.peerList = pl
		fmt.Printf("updated to %d peers: %s\n", len(pl), pl)
	}
}

func main() {
	flag.Parse()
	h := &configServer{}

	if len(*initFile) > 0 {
		f, err := os.Open(*initFile)
		if err != nil {
			utils.ExitErr(err)
		}
		defer f.Close()
		var pl plan.PeerList
		if err := readJSON(f, &pl); err != nil {
			utils.ExitErr(err)
		}
		h.peerList = pl
	}

	fmt.Printf("http://127.0.0.1:%d/get\n", *port)
	fmt.Printf("http://127.0.0.1:%d/put\n", *port)

	http.HandleFunc("/get", http.HandlerFunc(h.getConfig))
	http.HandleFunc("/put", http.HandlerFunc(h.putConfig))
	http.ListenAndServe(fmt.Sprintf(":%d", *port), nil)
}

func readJSON(r io.Reader, i interface{}) error {
	d := json.NewDecoder(r)
	return d.Decode(&i)
}
