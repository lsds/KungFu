package main

import (
	"os"
    "bytes"
    "encoding/json"
    "net/http"
    "strconv"
    "net"
     context"
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
    contentType := "application/json;charset=utf-8"
    data := "begin:" + strconv.Itoa(GoKungfuRank())
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
//export GoKungfuRunSendEnd
func GoKungfuRunSendEnd() {
    contentType := "application/json;charset=utf-8"
    data := "end:" + strconv.Itoa(GoKungfuRank())
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
//export GoKungfuRunSendEpoch
func GoKungfuRunSendEpoch() {
    contentType := "application/json;charset=utf-8"
    data := "epoch:" + strconv.Itoa(GoKungfuRank())
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
//export GoKungfuRunSendTrainend
func GoKungfuRunSendTrainend() {
    contentType := "application/json;charset=utf-8"
    data := "trainend:" + strconv.Itoa(GoKungfuRank())
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
