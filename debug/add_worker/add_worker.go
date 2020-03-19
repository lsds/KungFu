package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	configServer = flag.String("server", "http://127.0.0.1:9100/", "")
	period       = flag.Duration("period", 1*time.Second, "")
	ttl          = flag.Duration("ttl", 0, "time to live")
	reset        = flag.Bool("reset", false, "reset config server")
	clear        = flag.Bool("clear", false, "set peer list to empty")
	hostlist     = flag.String("H", "127.0.0.1:4", "")
)

func readJSON(r io.Reader, i interface{}) error {
	d := json.NewDecoder(r)
	return d.Decode(&i)
}

func main() {
	flag.Parse()
	u, err := url.Parse(*configServer)
	if err != nil {
		utils.ExitErr(err)
	}
	ctx := context.Background()
	if *ttl > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *ttl)
		defer cancel()
	}

	initFileName := "addWorkers.json"
	var f *os.File
	f, err = os.Open(initFileName)
	if err != nil {
		utils.ExitErr(err)
	}
	defer f.Close()
	var peers []plan.PeerID
	if err := readJSON(f, &peers); err != nil {
		utils.ExitErr(err)
	}

	for _, peer := range peers {
		addWorker(peer, *u)
		time.Sleep(1 * time.Minute)
	}
}

var client = http.Client{
	Timeout: 1 * time.Second,
}

func addWorker(peer plan.PeerID, endpoint url.URL) {
	reqBody, err := json.Marshal(peer)
	if err != nil {
		fmt.Println("Cannot marshal peer list")
	}
	endpoint.Path = `/addworker`
	resp, err := client.Post(endpoint.String(), "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		fmt.Printf("Cannot post request %v\n", err)
		return
	}
	if resp.StatusCode != http.StatusOK {
		fmt.Printf("%s\n", resp.Status)
	} else {
		fmt.Printf("OK\n")
	}
}
