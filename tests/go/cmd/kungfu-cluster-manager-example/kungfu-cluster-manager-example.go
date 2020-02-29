package main

import (
	"context"
	"flag"
	"fmt"
	"sync"
	"time"
)

var (
	image = flag.String("image", "kungfu-ci-base:snapshot", "")
	ttl   = flag.Duration("ttl", 120*time.Second, "")
)

func main() {
	flag.Parse()
	c := &cluster{
		image: *image,
		vnet:  `kungfu`,
		pool: &ipPool{
			prefix: `172.16.238`,
			n:      10,
		},
	}
	clear(c)
	example(c)
}

func example(c *cluster) {
	ctx := context.Background()
	c.Setup()

	wg := &sync.WaitGroup{}
	server := c.Start(ctx, wg, `kf-config-server`, `kungfu-peerlist-server`,
		`-ttl`, ttl.String(),
	)
	/*
		c.Start(ctx, wg, `kf-config-client`, `kungfu-peerlist-client`,
			`-server`, fmt.Sprintf("http://%s:%d/put", server.ip, 9100),
			`-ttl`, ttl.String(),
		)
	*/
	startWorker := func(name string) {
		c.Start(ctx, wg, name, `kungfu-run`,
			`-q`,
			`-w`,
			`-config-server`, fmt.Sprintf("http://%s:%d/get", server.ip, 9100),
			`-np`, `1`,
			`kungfu-fake-adaptive-trainer`,
		)

	}
	startWorker(`kf-node-01`)
	wg.Wait()
	c.Teardown()
}

func clear(c *cluster) {
	c.Remove(`kf-config-server`)
	c.Remove(`kf-config-client`)
	c.Teardown()
}
