package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/lsds/KungFu/srcs/go/cmd/kungfu-run/app"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
)

import "C"

var serverip string

type Message struct {
	Key string `json:"key"`
}

var httpc = http.Client{
	Transport: &http.Transport{
		DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
			peers := defaultPeer.CurrentSession().Peers()
			serverip = strings.Split(peers.String(), ":")[0]
			return net.Dial("tcp", serverip+":7756")
		},
	},
}

//export GoKungfuRunMain
func GoKungfuRunMain() {
	args := os.Args[1:] // remove wrapper program name (`which python`)
	app.Main(args)
}

//export GoKungfuSignalSend
func GoKungfuSignalSend(signal int) {
	contentType := "application/json;charset=utf-8"
	data := strconv.Itoa(GoKungfuRank())
	if signal == 1 {
		data = "begin:" + data
	} else if signal == 2 {
		data = "end:" + data
	} else if signal == 4 {
		data = "epoch:" + data
	} else if signal == 3 {
		data = "trainend:" + data
	}
	msg := Message{Key: data}
	b, err := json.Marshal(msg)
	if err != nil {
		return
	}
	body := bytes.NewBuffer(b)
	resp, err := httpc.Post("http://"+serverip+":7756", contentType, body)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer resp.Body.Close()
}
