package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	image = flag.String("image", "kungfu-ci-base:snapshot", "")
	ttl   = flag.Duration("ttl", 120*time.Second, "")
)

var errMissingProgramName = errors.New("missing program name")

func main() {
	flag.Parse()
	args := flag.Args()
	if len(args) < 1 {
		utils.ExitErr(errMissingProgramName)
	}
	c := &cluster{
		image: *image,
		vnet:  `kungfu`,
		pool: &ipPool{
			prefix: `172.16.238`,
			n:      10,
		},
	}
	clear(c)
	example(c, args[0], args[1:])
}

func example(c *cluster, prog string, args []string) {
	ctx := context.Background()
	c.Setup()

	wg := &sync.WaitGroup{}
	const configServerPort = 9100
	server := c.Start(ctx, wg, `kf-config-server`,
		proc{
			cmd: `kungfu-config-server-example`,
			args: []string{
				`-ttl`, ttl.String(),
			},
			port: configServerPort,
		},
	)
	/*
		c.Start(ctx, wg, `kf-config-client`, `kungfu-config-client-example`,
			`-server`, fmt.Sprintf("http://%s:%d/put", server.ip, 9100),
			`-ttl`, ttl.String(),
		)
	*/

	getConfigURL := fmt.Sprintf("http://%s:%d/get", server.ip, configServerPort)
	putConfigURL := fmt.Sprintf("http://%s:%d/put", `127.0.0.1`, configServerPort)

	cc := &configClient{
		endpoint: putConfigURL,
	}
	cc.Wait()

	var hl plan.HostList

	startWorker := func(name string, cap int, isFirst bool) {
		ip := c.pool.get()
		hl = plan.HostList{
			{
				IPv4:  plan.MustParseIPv4(ip),
				Slots: cap,
			},
		}
		pl, err := hl.GenPeerList(hl.Cap(), plan.DefaultPortRange)
		if err != nil {
			utils.ExitErr(err)
			return
		}
		cluster := plan.Cluster{
			Runners: hl.GenRunnerList(plan.DefaultRunnerPort),
			Workers: pl,
		}
		if err := cc.Update(cluster); err != nil {
			utils.ExitErr(err)
			return
		}
		initVersion := -1
		if isFirst {
			initVersion = 0
		}
		kungfuRunArgs := []string{
			`-q`,
			`-timeout`, ttl.String(),
			`-H`, hl.String(),
			`-self`, ip,
			`-w`,
			`-config-server`, getConfigURL,
			`-init-version`, strconv.Itoa(initVersion),
			prog,
		}
		c.StartWithIP(ctx, wg, name, ip,
			proc{
				cmd:  `kungfu-run`,
				args: append(kungfuRunArgs, args...),
			},
		)
	}
	startWorker(`kf-node-01`, 4, true)
	startWorker(`kf-node-02`, 4, false)
	wg.Wait()
	c.Teardown()
}

func clear(c *cluster) {
	c.Remove(`kf-config-server`)
	c.Remove(`kf-config-client`)
	c.Teardown()
}
