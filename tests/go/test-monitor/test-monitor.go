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
)

var (
	port     = flag.Int("port", 8080, "")
	duration = flag.Duration("d", 1*time.Second, "")
	period   = flag.Duration("p", 10*time.Millisecond, "")
)

func main() {
	flag.Parse()
	go monitor.ListenAndServe(*port)

	var wg sync.WaitGroup
	wg.Add(2)

	t0 := time.Now()

	go func() {
		nm := monitor.GetNetMetrics()
		tk := time.NewTicker(*period)
		defer tk.Stop()
		var i int64
		for t := range tk.C {
			i = (i*10007 + 17) % 97
			nm.Sent(i)
			nm.Recv(i)
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
