package main

import (
	"context"
	"flag"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/hostfile"
	"github.com/lsds/KungFu/srcs/go/proc"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
	"github.com/lsds/KungFu/srcs/go/utils/runner/remote"
)

var flg = struct {
	hostfile   *string
	usr        *string
	logDir     *string
	tag        *string
	python     *string
	enableNCCL *bool
}{
	hostfile:   flag.String("hostfile", "hosts.txt", ""),
	usr:        flag.String("u", "", "user name for ssh"),
	logDir:     flag.String("logdir", ".", ""),
	tag:        flag.String("tag", "master", ""),
	enableNCCL: flag.Bool("nccl", false, ""),
	python:     flag.String("python", "python3", ""),
}

func main() {
	flag.Parse()
	t0 := time.Now()
	defer func(prog string) { log.Infof("%s finished, took %s", prog, time.Since(t0)) }(utils.ProgName())
	log.Infof("%s started", utils.ProgName())

	hl, err := hostfile.ParseFile(*flg.hostfile)
	if err != nil {
		utils.ExitErr(err)
	}
	if err := installAll(hl); err != nil {
		utils.ExitErr(err)
	}
}

var str = strconv.Itoa

func b2i(f bool) int {
	if f {
		return 1
	}
	return 0
}

func installAll(hl plan.HostList) error {
	const kfDir = `.kungfu/KungFu`

	cmds := shellCmds{
		// parseCmd(`mkdir -p local`),
		// parseCmd(`curl -sLOJ https://dl.google.com/go/go1.14.3.linux-amd64.tar.gz`).ChDir(`local`),
		// parseCmd(`tar -xvf go1.14.3.linux-amd64.tar.gz`).ChDir(`local`),

		parseCmd(`rm -fr .kungfu`),
		parseCmd(`mkdir -p .kungfu`),
		parseCmd(`ls .kungfu`),
		parseCmd(`git clone https://github.com/lsds/KungFu ` + kfDir),
		parseCmd(`git checkout ` + *flg.tag).ChDir(kfDir),
		parseCmd(`go install -v ./...`).ChDir(kfDir).
			Env(`PATH`, `$HOME/local/go/bin:$PATH`),
		parseCmd(*flg.python+` -m pip install --user --no-index -U .`).
			ChDir(kfDir).
			Env(`PATH`, `$HOME/local/go/bin:$PATH`).
			Env(`KUNGFU_ENABLE_NCCL`, str(b2i(*flg.enableNCCL))).
			Env(`KUNGFU_BUILD_TOOLS`, str(0)),
	}

	var wg sync.WaitGroup
	for _, h := range hl {
		wg.Add(1)
		go func(hostname string) {
			waitSSH(hostname)
			cmds.RunOn(hostname)
			wg.Done()
		}(h.PublicAddr)
	}
	wg.Wait()

	return nil
}

type shellCmd struct {
	prog  string
	args  []string
	envs  proc.Envs
	chdir string
}

func (s shellCmd) Env(k, v string) shellCmd {
	if s.envs == nil {
		s.envs = make(proc.Envs)
	}
	s.envs[k] = v
	return s
}

func (s shellCmd) ChDir(dir string) shellCmd {
	s.chdir = dir
	return s
}

func parseCmd(line string) shellCmd {
	parts := strings.Split(line, " ")
	assert.True(len(parts) > 0)
	return shellCmd{prog: parts[0], args: parts[1:]}
}

type shellCmds []shellCmd

func (ss shellCmds) RunOn(hostname string) error {
	for i, c := range ss {
		p := proc.Proc{
			Name:     fmt.Sprintf("%s#%d$%s", hostname, i, c.prog),
			Prog:     c.prog,
			Args:     c.args,
			Hostname: hostname,
			Envs:     c.envs,
			Dir:      c.chdir,
		}
		log.Infof("running on %s $ %s %q", p.Hostname, p.Prog, p.Args)
		if err := remote.RemoteRunAll(context.TODO(), *flg.usr, []proc.Proc{p}, true, *flg.logDir); err != nil {
			log.Errorf("failed to run %s $ %s %q: %v", p.Hostname, p.Prog, p.Args, err)
			return err
		}
	}
	return nil
}

func waitSSH(hostname string) {
	p := proc.Proc{
		Name:     fmt.Sprintf("%s$%s", hostname, `wait-ssh`),
		Prog:     `pwd`,
		Hostname: hostname,
	}
	trial := func() bool {
		err := remote.RemoteRunAll(context.TODO(), *flg.usr, []proc.Proc{p}, true, *flg.logDir)
		if err != nil {
			log.Warnf("still waiting %s", hostname)
		}
		return err == nil
	}
	poll(trial, 1*time.Second)
}

func poll(f func() bool, duration time.Duration) {
	for {
		if ok := f(); ok {
			return
		}
		time.Sleep(duration)
	}
}
