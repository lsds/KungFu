package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/client"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	kill         = flag.String("kill", "", "peer id to terminate")
	propose      = flag.String("propose", "", "path to new config file")
	configServer = flag.String("server", "http://127.0.0.1:9100/", "")
)

func main() {
	flag.Parse()
	switch {
	case len(*kill) > 0:
		target, err := plan.ParsePeerID(*kill)
		if err != nil {
			utils.ExitErr(err)
		}
		terminate(*target)
		return
	case len(*propose) > 0:
		f, err := os.Open(*propose)
		if err != nil {
			utils.ExitErr(err)
		}
		defer f.Close()
		var cluster plan.Cluster
		if err := json.NewDecoder(f).Decode(&cluster); err != nil {
			utils.ExitErr(err)
		}
		u, err := url.Parse(*configServer)
		if err != nil {
			utils.ExitErr(err)
		}
		postConfig(cluster, *u)
		return
	default:
		flag.Usage()
		utils.ShowBuildInfo()
	}
}

func terminate(target plan.PeerID) {
	client := client.New(plan.PeerID{}, false) // FIXME: don't retry connect
	if err := client.Send(target.WithName("exit"), nil, connection.ConnControl, connection.NoFlag); err != nil {
		log.Errorf("failed to send exit signal sent to %s", target)
		return
	}
	log.Infof("exit signal sent to %s", target)
}

func postConfig(clustr plan.Cluster, endpoint url.URL) {
	client := http.Client{
		Timeout: 1 * time.Second,
	}
	reqBody, err := json.Marshal(clustr)
	if err != nil {
		fmt.Println("Cannot marshal peer list")
	}
	endpoint.Path = `/put`
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
