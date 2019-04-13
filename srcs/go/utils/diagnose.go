package utils

import (
	"context"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
)

const magic = "OK"

func TestConnectivity(cs *plan.ClusterSpec, idx int) error {
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	client := &http.Client{
		Timeout: 1 * time.Second,
	}
	myAddr := cs.Peers[idx].NetAddr
	myName := fmt.Sprintf("peer-%d@%s:%d", idx, myAddr.Host, myAddr.Port)
	var wg sync.WaitGroup
	var fail int32
	for i, p := range cs.Peers {
		if i != idx {
			wg.Add(1)
			go func(i int, p plan.PeerSpec) {
				url := fmt.Sprintf("http://%s:%d/from/%s", p.NetAddr.Host, p.NetAddr.Port, myName)
				if err := waitOK(ctx, myName, client, url); err != nil {
					atomic.AddInt32(&fail, 1)
				}
				wg.Done()
			}(i, p)
		}
	}
	server := http.Server{
		Addr: net.JoinHostPort("0.0.0.0", strconv.Itoa(int(myAddr.Port))),
		Handler: http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			fmt.Printf("[diagnose] peer-%d@%s:%d <- %s %s %s\n", idx, myAddr.Host, myAddr.Port, req.Method, req.URL, req.RemoteAddr)
			w.Write([]byte(magic))
		}),
	}
	go func() { server.ListenAndServe() }()
	defer server.Close()
	wg.Wait()
	if fail > 0 {
		return fmt.Errorf("can't connect to %d other peers", fail)
	}
	fmt.Printf("[diagnose] PASS\n")
	return nil
}

func waitOK(ctx context.Context, myName string, client *http.Client, url string) error {
	tk := time.NewTicker(1 * time.Second)
	defer tk.Stop()
	for {
		select {
		case <-tk.C:
			resp, err := client.Get(url)
			if err != nil {
				fmt.Printf("[diagnose] %s %v\n", myName, err)
				continue
			}
			defer resp.Body.Close()
			bs, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				fmt.Printf("[diagnose] %s %v\n", myName, err)
				continue
			}
			if string(bs) == magic {
				fmt.Printf("[diagnose] %s can access %s\n", myName, url)
				return nil
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}
