package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
)

var (
	port     = flag.Int("port", 9999, "")
	duration = flag.Duration("d", 1*time.Second, "")
	period   = flag.Duration("p", 10*time.Millisecond, "")
)

func main() {
	flag.Parse()
	monitor.StartServer(*port)
	defer monitor.StopServer()

	var wg sync.WaitGroup
	wg.Add(2)

	t0 := time.Now()

	go func() {
		mon := monitor.GetMonitor()
		tk := time.NewTicker(*period)
		addrs := genTestAddrs()
		defer tk.Stop()
		var i int
		for t := range tk.C {
			i++
			j := int64((i*10007 + 17) % 97)
			a := addrs[i%len(addrs)]
			mon.Egress(j, a)
			mon.Ingress(j, a)
			if t.Sub(t0) > *duration {
				break
			}
		}
		wg.Done()
	}()

	go func() {
		c := newClient(fmt.Sprintf(`http://127.0.0.1:%d/metrics`, *port))
		tk := time.NewTicker(*period)
		defer tk.Stop()
		for t := range tk.C {
			fmt.Printf("%s\n", t)
			c.getMetrics()
			if t.Sub(t0) > *duration {
				break
			}
		}
		wg.Done()
	}()
	wg.Wait()
}

func genTestAddrs() []plan.NetAddr {
	var addrs []plan.NetAddr
	for i := 0; i < 10; i++ {
		addrs = append(addrs, plan.NetAddr{
			IPv4: plan.MustParseIPv4(`127.0.0.1`),
			Port: uint16(9999 + i),
		})
	}
	return addrs
}

type client struct {
	endpoint string
	client   *http.Client
}

func newClient(url string) *client {
	return &client{
		endpoint: url,
		client:   &http.Client{},
	}
}

func (c client) getMetrics() error {
	resp, err := c.client.Get(c.endpoint)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, err = io.Copy(os.Stdout, resp.Body)
	return err
}
