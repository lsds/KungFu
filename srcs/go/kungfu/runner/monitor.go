package runner

import (
	"encoding/json"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type monitorserver struct {
	DownFlag   bool
	Machines   int
	Epochnum   int
	FinishFlag bool
	trainend   []int
	times      []int64
	epochs     []int
	wg         sync.WaitGroup
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
}
func (s *monitorserver) Start() {

	defer s.wg.Done()
	for i := 0; i < s.Machines; i++ {
		s.trainend = append(s.trainend, 0)
		s.times = append(s.times, 0)
		s.epochs = append(s.epochs, 0)
	}
	_, err := os.Stat("/tmp/http.sock")
	if err != nil && os.IsNotExist(err) {
	} else {
		err1 := os.Remove("/tmp/http.sock")
		if err1 != nil {
			panic("Cannot delete: " + err.Error())
		}
	}

	addr, err := net.ResolveUnixAddr("unix", "/tmp/http.sock")
	if err != nil {
		panic("Cannot resolve unix addr: " + err.Error())
	}
	listener, err := net.ListenUnix("unix", addr)
	if err != nil {
		panic("Cannot resolve unix addr: " + err.Error())
	}
	go http.Serve(listener, s)
	for {
		trainendflag := 0
		downflag := false
		for i := 0; i < s.Machines; i++ {
			if s.trainend[i] == 1 {
				trainendflag = trainendflag + 1
			}
			if a := time.Now().Unix() - s.times[i]; a > 10 {
				if s.times[i] != 0 {
					min := findmin(s.epochs)
					s.DownFlag = true
					s.Epochnum = min
					downflag = true
					break
				}
			}
		}
		if downflag {
			break
		}
		if trainendflag == s.Machines {
			s.FinishFlag = true
			break
		}
	}
	listener.Close()
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
