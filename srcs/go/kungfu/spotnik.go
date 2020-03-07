package kungfu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func timeoutHelper(timeoutDuration time.Duration, op func(), timeoutCallback func()) {
	ch := make(chan bool, 1)
	go func() {
		op()
		ch <- true
	}()
	select {
	case <-ch:
	case <-time.After(timeoutDuration):
		log.Errorf("timed out")
		timeoutCallback()
	}
}

func healthCheck(self plan.PeerID, target plan.PeerID, configServer string) {
	conn, err := rch.NewPingConnection(plan.NetAddr(target), plan.NetAddr(self))
	if err != nil {
		log.Errorf("ping failed %s -> %s", plan.NetAddr(self), plan.NetAddr(target))
		removeWorker(target, configServer)
	}
	if conn != nil {
		defer conn.Close()
	}
}

func removeWorker(worker plan.PeerID, configServer string) {
	client := http.Client{Timeout: 1 * time.Second}
	endpoint, err := url.Parse(configServer)
	if err != nil {
		utils.ExitErr(err)
	}
	reqBody, err := json.Marshal(worker)
	if err != nil {
		fmt.Println("Cannot marshal peer list")
	}
	endpoint.Path = "/removeworker"
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
