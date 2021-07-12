package monitorserver

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/lsds/KungFu/srcs/go/plan"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

type monitorserver struct {
	DownFlag      bool
	Machines      int
	Epochnum      int
	FinishFlag    bool
	trainend      []int
	times         []int64
	epochs        []int
	wg            sync.WaitGroup
	OtherFinish   bool
	OtherEpochnum int
	OtherDown     bool
	serverip      string
}
type Results struct {
	DownFlag   bool
	Epochnum   int
	FinishFlag bool
}

var trainend []int
var times []int64
var epochs []int

type Message struct {
	Key string `json:"key"`
}

func (h *monitorserver) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var msg Message
	err := json.NewDecoder(r.Body).Decode(&msg)
	if err != nil {
		return
	}
	datas := strings.Split(string(msg.Key), ":")
	intva, err := strconv.Atoi(datas[1])
	if err != nil {
	}
	if datas[0] == "trainend" {
		h.trainend[intva] = 1
	}
	if datas[0] == "begin" {
		h.times[intva] = time.Now().Unix()
	}
	if datas[0] == "end" {
		h.times[intva] = 0
	}
	if datas[0] == "epoch" {
		h.epochs[intva] = h.epochs[intva] + 1
	}
	if datas[0] == "otherfinish" {
		h.OtherFinish = true
	}
	if datas[0] == "otherdown" {
		h.OtherEpochnum = intva
		h.OtherDown = true
	}
}
func (s *monitorserver) Start(ip string, hl plan.HostList, ClusterSize int, waittime int) {
	var otherips []string
	isMainServer := false
	ipnum := 0
	if len(hl) != 1 {
		for _, h := range hl {
			if plan.FormatIPv4(h.IPv4) == ip {
				if ipnum == 0 {
					isMainServer = true
				}
			} else {
				otherips = append(otherips, plan.FormatIPv4(h.IPv4))
			}
			ipnum = ipnum + 1
		}
	}
	server := &http.Server{
		Addr:    ip + ":7756",
		Handler: s,
	}
	defer s.wg.Done()
	for i := 0; i < ClusterSize; i++ {
		s.trainend = append(s.trainend, 0)
		s.times = append(s.times, 0)
		s.epochs = append(s.epochs, 0)
	}
	go func() {
		server.ListenAndServe()
		s.wg.Done()
	}()
	for {
		time.Sleep(1)
		trainendflag := 0
		downflag := false
		for i := 0; i < ClusterSize; i++ {
			if s.trainend[i] == 1 {
				trainendflag = trainendflag + 1
			}
			if a := time.Now().Unix() - s.times[i]; a > int64(10) && s.times[i] != 0 {
				min := findmin(s.epochs)
				s.DownFlag = true
				s.Epochnum = min
				downflag = true
				if isMainServer {
					contentType := "application/json;charset=utf-8"
					data := "otherdown:" + strconv.Itoa(min)
					msg := Message{Key: data}
					b, err := json.Marshal(msg)
					if err != nil {
						return
					}
					body := bytes.NewBuffer(b)
					for _, otherip := range otherips {
						resp, err := http.Post("http://"+otherip+":7756", contentType, body)
						if err != nil {
							return
						}
						defer resp.Body.Close()
					}
				}
				break
			}
		}
		if s.OtherDown {
			s.DownFlag = true
			s.Epochnum = s.OtherEpochnum
			break
		}
		if downflag {
			break
		}

		if trainendflag == ClusterSize || s.OtherFinish {
			if isMainServer {
				contentType := "application/json;charset=utf-8"
				data := "otherfinish:0"
				msg := Message{Key: data}
				b, err := json.Marshal(msg)
				if err != nil {
					return
				}
				body := bytes.NewBuffer(b)
				for _, otherip := range otherips {
					resp, err := http.Post("http://"+otherip+":7756", contentType, body)
					if err != nil {
						fmt.Println(err)

						return
					}
					defer resp.Body.Close()
				}
			}
			s.FinishFlag = true
			break
		}
	}
	server.Shutdown(context.TODO())
}
func findmin(array []int) int {
	min := array[0]
	for _, v := range array {
		if v < min {
			min = v
		}
	}
	return min
}

func (s *monitorserver) Wait() Results {
	s.wg.Wait()
	return Results{
		DownFlag:   s.DownFlag,
		Epochnum:   s.Epochnum,
		FinishFlag: s.FinishFlag,
	}
}

func New(procs int) *monitorserver {
	return &monitorserver{Machines: procs}
}

func (s *monitorserver) Monitor(ip string, hl plan.HostList, ClusterSize int, waittime int) {
	s.wg.Add(2)
	go s.Start(ip, hl, ClusterSize, waittime)

}
