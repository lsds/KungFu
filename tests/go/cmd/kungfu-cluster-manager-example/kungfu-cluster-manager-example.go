package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/tests/go/configserver"
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
			cmd: `kungfu-config-server`,
			args: []string{
				`-ttl`, ttl.String(),
			},
			port: configServerPort,
		},
	)

	configURL := fmt.Sprintf("http://%s:%d/config", server.ip, configServerPort)
	putConfigURL := fmt.Sprintf("http://%s:%d/config", `127.0.0.1`, configServerPort)
	cc := configserver.NewClient(putConfigURL)
	cc.WaitServer()

	var hl plan.HostList

	var workerGroup sync.WaitGroup
	startWorker := func(name string, cap int, isFirst bool) *node {
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
			return nil
		}
		cluster := plan.Cluster{
			Runners: hl.GenRunnerList(plan.DefaultRunnerPort),
			Workers: pl,
		}
		if err := cc.Update(cluster); err != nil {
			utils.ExitErr(err)
			return nil
		}
		delay := `10s`
		initVersion := -1
		if isFirst {
			initVersion = 0
			delay = `0s`
		}
		kungfuRunArgs := []string{
			`-q`,
			`-timeout`, ttl.String(),
			`-H`, hl.String(),
			`-self`, ip,
			`-w`,
			`-config-server`, configURL,
			`-init-version`, strconv.Itoa(initVersion),
			`-delay`, delay,
			prog,
		}
		node := c.StartWithIP(ctx, wg, name, ip,
			proc{
				cmd:  `kungfu-run`,
				args: append(kungfuRunArgs, args...),
			},
		)
		workerGroup.Add(1)
		go func() {
			node.Wait()
			workerGroup.Done()
		}()
		return node
	}

	startWorker(`kf-node-01`, 4, true)
	time.Sleep(5 * time.Second)
	startWorker(`kf-node-02`, 4, false)

	log.Infof("waiting all workers to stop")
	workerGroup.Wait()
	log.Infof("all nodes stopped, stopping config server")
	cc.StopServer()

	wg.Wait()
	c.Teardown()
}

func clear(c *cluster) {
	c.Remove(`kf-config-server`)
	c.Teardown()
}
