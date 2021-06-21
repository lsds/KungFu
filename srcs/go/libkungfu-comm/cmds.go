package main

import (
	"os"
    "bytes"
    "encoding/json"
    "net/http"
    "strconv"
    "net"
    "context"
	"github.com/lsds/KungFu/srcs/go/cmd/kungfu-run/app"
)

import "C"
type Message struct {
    Key string `json:"key"`
}
var httpc = http.Client{
        Transport: &http.Transport{
            DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
                return net.Dial("unix","/tmp/http.sock")
            },
        },
    }
//export GoKungfuRunMain
func GoKungfuRunMain() {
	args := os.Args[1:] // remove wrapper program name (`which python`)
	app.Main(args)
}

//export GoKungfuRunSendBegin
func GoKungfuRunSendBegin() {
    SignalSend(1)
}
//export GoKungfuRunSendEnd
func GoKungfuRunSendEnd() {
    SignalSend(2)
}
//export GoKungfuRunSendEpoch
func GoKungfuRunSendEpoch() {
    SignalSend(3)
}
//export GoKungfuRunSendTrainend
func GoKungfuRunSendTrainend() {
    SignalSend(4)
}

func SignalSend(signal int) {
    contentType := "application/json;charset=utf-8"
    data := strconv.Itoa(GoKungfuRank())
    if signal == 1{
        data = "begin:" + data
    } else if signal == 2{
        data = "end:" + data
    } else if signal == 3{
        data = "epoch:" + data
    } else if signal == 4{
        data = "trainend:" + data
    }
    msg := Message{Key: data}
    b ,err := json.Marshal(msg)
    if err != nil {
        return
    }
    body := bytes.NewBuffer(b)
    resp, err := httpc.Post("http://http.sock",contentType,body)
    if err != nil {
        return
    }
    defer resp.Body.Close()
}
