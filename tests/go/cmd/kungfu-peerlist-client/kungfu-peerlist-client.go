package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	configServer = flag.String("server", "http://127.0.0.1:9100/put", "")
	period       = flag.Duration("period", 1*time.Second, "")
	clear        = flag.Bool("clear", false, "set peer list to empty")
)

func main() {
	flag.Parse()
	if *clear {
		clearConfig()
		return
	}
	periodically(*period, updateConfig)
}

func postConfig(pl plan.PeerList) {
	reqBody, err := json.Marshal(pl)
	if err != nil {
		fmt.Println("Cannot marshal peer list")
	}
	_, err = http.Post(*configServer, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		fmt.Printf("Cannot post request %v\n", err)
		return
	}
	fmt.Printf("OK\n")
}

func clearConfig() {
	pl := plan.PeerList{}
	postConfig(pl)
}

func updateConfig() {
	hl := plan.HostList{
		{
			IPv4:  plan.MustParseIPv4(`127.0.0.1`),
			Slots: 8,
		},
	}
	pl := genPeerList(hl)
	fmt.Printf("updating to %d peers: %s\n", len(pl), pl)
	postConfig(pl)
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
