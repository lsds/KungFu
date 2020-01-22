package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"sync"

	"github.com/lsds/KungFu/srcs/go/plan"
)

type configServer struct {
	sync.RWMutex

	peerList plan.PeerList
}

func (s *configServer) getConfig(w http.ResponseWriter, req *http.Request) {
	s.RLock()
	defer s.RUnlock()
	jsonPeerList, err := json.Marshal(s.peerList)
	if err != nil {
		fmt.Println("Cannot marshal peer list")
	}
	fmt.Fprintf(w, "%s\n", jsonPeerList)
}

func (s *configServer) putConfig(w http.ResponseWriter, req *http.Request) {
	s.Lock()
	defer s.Unlock()

	body, err := ioutil.ReadAll(req.Body)
	if err != nil {
		fmt.Println("Cannot read request body")
	}

	fmt.Println(string(body))

	err = json.Unmarshal(body, &s.peerList)
	if err != nil {
		fmt.Println("Cannot unmarshal body")
	}

	fmt.Fprintf(w, "OK\n")
}

var port = flag.Int("port", 9100, "")
var peerlistPath = flag.String("path", "./hostlists/hostlist.json", "")

func main() {
	flag.Parse()
	h := &configServer{}

	content, err := ioutil.ReadFile(*peerlistPath)
	if err != nil {
		fmt.Println("Cannot read file")
	}
	err = json.Unmarshal(content, &h.peerList)
	if err != nil {
		fmt.Println("Cannot unmarshal content")
	}

	fmt.Printf("http://127.0.0.1:%d/get\n", *port)
	fmt.Printf("http://127.0.0.1:%d/put\n", *port)

	http.HandleFunc("/get", http.HandlerFunc(h.getConfig))
	http.HandleFunc("/put", http.HandlerFunc(h.putConfig))
	http.ListenAndServe(fmt.Sprintf(":%d", *port), nil)
}
