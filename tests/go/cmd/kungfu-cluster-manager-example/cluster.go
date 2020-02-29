package main

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"sync"

	"github.com/lsds/KungFu/srcs/go/log"
	runner "github.com/lsds/KungFu/srcs/go/utils/runner/local"
)

type cluster struct {
	image string
	vnet  string
	pool  *ipPool
}

type proc struct {
	cmd  string
	args []string
	port int
}

type node struct {
	ip   string
	name string
	wg   *sync.WaitGroup
}

func (c cluster) Setup() {
	run(`docker`, `network`, `create`, c.vnet, `--subnet`, c.pool.subnet())
}

func (c cluster) Teardown() {
	run(`docker`, `network`, `rm`, c.vnet)
}

func (c cluster) Start(ctx context.Context, all *sync.WaitGroup, name string, p proc) *node {
	ip := c.pool.get()
	return c.StartWithIP(ctx, all, name, ip, p)
}

func (c cluster) StartWithIP(ctx context.Context, all *sync.WaitGroup, name, ip string, p proc) *node {
	wg := &sync.WaitGroup{}
	all.Add(1)
	wg.Add(1)
	go func() {
		c.Run(ctx, name, ip, p)
		wg.Done()
		all.Done()
		log.Infof("%s @ %s stopped", name, ip)
	}()
	log.Infof("%s created: %s", name, ip)
	return &node{ip: ip, name: name, wg: wg}
}

func (c cluster) Run(ctx context.Context, name, ip string, p proc) {
	dockerArgs := []string{
		`run`, `--rm`,
		`--name`, name,
		`--network`, c.vnet,
		`--ip`, ip,
	}
	if p.port > 0 {
		dockerArgs = append(dockerArgs, `-p`, fmt.Sprintf("%d:%d", p.port, p.port))
	}
	dockerArgs = append(dockerArgs, `-t`, c.image)
	dockerArgs = append(dockerArgs, p.cmd)
	dockerArgs = append(dockerArgs, p.args...)
	run(`docker`, dockerArgs...)
}

func (c cluster) Remove(name string) {
	dockerArgs := []string{
		`rm`, `-f`,
		name,
	}
	run(`docker`, dockerArgs...)
}

func run(prog string, args ...string) error {
	log.Infof("$ %s %s", prog, strings.Join(args, " "))
	cmd := exec.Command(prog, args...)
	r := runner.Runner{
		Name:       "docker",
		LogDir:     "logs",
		VerboseLog: true,
	}
	return r.Run(context.TODO(), cmd)
}
