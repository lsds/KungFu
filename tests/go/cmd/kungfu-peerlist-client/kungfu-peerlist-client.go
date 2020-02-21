package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"net/url"
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	configServer = flag.String("server", "http://127.0.0.1:9100/", "")
	period       = flag.Duration("period", 1*time.Second, "")
	reset        = flag.Bool("reset", false, "reset config server")
	clear        = flag.Bool("clear", false, "set peer list to empty")
)

func main() {
	flag.Parse()
	u, err := url.Parse(*configServer)
	if err != nil {
		utils.ExitErr(err)
	}
	if *clear {
		clearConfig(*u)
		return
	}
	if *reset {
		resetConfig(*u)
		return
	}
	periodically(*period, func() { updateConfig(*u) })
}

func postConfig(pl plan.PeerList, endpoint url.URL) {
	reqBody, err := json.Marshal(pl)
	if err != nil {
		fmt.Println("Cannot marshal peer list")
	}
	endpoint.Path = `/put`
	resp, err := http.Post(endpoint.String(), "application/json", bytes.NewBuffer(reqBody))
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

func resetConfig(endpoint url.URL) {
	endpoint.Path = `/reset`
	http.Post(endpoint.String(), "application/json", nil)
}

func clearConfig(endpoint url.URL) {
	pl := plan.PeerList{}
	postConfig(pl, endpoint)
}

func updateConfig(endpoint url.URL) {
	hl := plan.HostList{
		{
			IPv4:  plan.MustParseIPv4(`127.0.0.1`),
			Slots: 8,
		},
	}
	pl := genPeerList(hl)
	fmt.Printf("updating to %d peers: %s\n", len(pl), pl)
	postConfig(pl, endpoint)
}

func genPeerList(hl plan.HostList) plan.PeerList {
	pl, err := hl.GenPeerList(hl.Cap(), plan.DefaultPortRange)
	if err != nil {
		utils.ExitErr(err)
	}
	n := rand.Int()%(len(pl)-1) + 1
	pl = pl[:n]
	rand.Shuffle(len(pl), func(i, j int) { pl[i], pl[j] = pl[j], pl[i] })
	return pl
}

func periodically(p time.Duration, f func()) {
	tk := time.NewTicker(p)
	defer tk.Stop()
	f()
	for range tk.C {
		f()
	}
}
