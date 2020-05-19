package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/runner"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	configServer = flag.String("server", "http://127.0.0.1:9100/", "")
	period       = flag.Duration("period", 1*time.Second, "")
	ttl          = flag.Duration("ttl", 0, "time to live")
	reset        = flag.Bool("reset", false, "reset config server")
	clear        = flag.Bool("clear", false, "set peer list to empty")
	example      = flag.Bool("example", false, "show example config")
	hostlist     = flag.String("H", "127.0.0.1:4", "")
)

func main() {
	t0 := time.Now()
	flag.Parse()
	hl, err := runner.ResolveHostList(*hostlist, "")
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to parse -H: %v", err))
	}
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
	if *example {
		showExample()
		return
	}
	ctx := context.Background()
	if *ttl > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *ttl)
		defer cancel()
	}
	periodically(ctx, *period, func(idx int) { updateConfig(*u, hl) })
	log.Infof("%s stopped after %s", utils.ProgName(), time.Since(t0))
}

var client = http.Client{
	Timeout: 1 * time.Second,
}

func postConfig(clustr plan.Cluster, endpoint url.URL) {
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

func resetConfig(endpoint url.URL) {
	endpoint.Path = `/reset`
	client.Post(endpoint.String(), "application/json", nil)
}

func clearConfig(endpoint url.URL) {
	clustr := plan.Cluster{}
	postConfig(clustr, endpoint)
}

func updateConfig(endpoint url.URL, hl plan.HostList) {
	pl := genPeerList(hl)
	fmt.Printf("updating to %d peers: %s\n", len(pl), pl)
	clustr := plan.Cluster{
		Runners: hl.GenRunnerList(plan.DefaultRunnerPort),
		Workers: pl,
	}
	postConfig(clustr, endpoint)
}

func genPeerList(hl plan.HostList) plan.PeerList {
	pl, err := hl.GenPeerList(hl.Cap(), plan.DefaultPortRange)
	if err != nil {
		utils.ExitErr(err)
	}
	rand.Shuffle(len(pl), func(i, j int) { pl[i], pl[j] = pl[j], pl[i] })
	n := rand.Int()%len(pl) + 1
	pl = pl[:n]
	rand.Shuffle(len(pl), func(i, j int) { pl[i], pl[j] = pl[j], pl[i] })
	return pl
}

func periodically(ctx context.Context, p time.Duration, f func(int)) {
	tk := time.NewTicker(p)
	defer tk.Stop()
	var idx int
	f(idx)
	for {
		select {
		case <-tk.C:
			idx++
			f(idx)
		case <-ctx.Done():
			return
		}
	}
}

func showExample() {
	hl := plan.HostList{
		{
			IPv4:  plan.MustParseIPv4(`192.168.1.11`),
			Slots: 4,
		},
		{
			IPv4:  plan.MustParseIPv4(`192.168.1.10`),
			Slots: 4,
		},
	}
	pl, err := hl.GenPeerList(hl.Cap(), plan.DefaultPortRange)
	if err != nil {
		utils.ExitErr(err)
	}
	c := plan.Cluster{
		Runners: hl.GenRunnerList(plan.DefaultRunnerPort),
		Workers: pl,
	}
	e := json.NewEncoder(os.Stdout)
	e.SetIndent("", "    ")
	e.Encode(c)
}
