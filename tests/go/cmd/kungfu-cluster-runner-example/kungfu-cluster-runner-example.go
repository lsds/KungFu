package main

import (
	"context"
	"errors"
	"flag"
	"strconv"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
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

	var workerGroup sync.WaitGroup
	startWorker := func(nums int, cap int) {
		hall := ""
		ips := []string{}
		for i := 0; i < nums; i++ {
			ip := c.pool.get()
			ips = append(ips, ip)
			iph := ip + ":" + strconv.Itoa(cap)
			if hall == "" {
				hall = iph
			} else {
				hall = hall + "," + iph
			}
		}
		for _, ip := range ips {
			kungfuRunArgs := []string{
				`-np`, strconv.Itoa(nums * cap),
				`-mnt`, `10`,
				`-H`, hall,
				//`-self`, ip,
				`-nic`, `eth0`,
				prog,
			}
			node := c.StartWithIP(ctx, wg, ip, ip,
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
		}
	}

	startWorker(2, 1)

	log.Infof("waiting all workers to stop")
	workerGroup.Wait()
	log.Infof("all nodes stopped, stopping config server")

	wg.Wait()
	c.Teardown()
}

func clear(c *cluster) {
	c.Remove(`kf-config-server`)
	c.Teardown()
}
