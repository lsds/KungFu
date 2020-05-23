package main

import (
	"context"
	"flag"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/hostfile"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
	"github.com/lsds/KungFu/srcs/go/utils/runner/remote"
)

var flg = struct {
	hostfile *string
	usr      *string
	logDir   *string
}{
	hostfile: flag.String("hostfile", "hosts.txt", ""),
	usr:      flag.String("u", "", "user name for ssh"),
	logDir:   flag.String("logdir", ".", ""),
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

func installAll(hl plan.HostList) error {
	cmds := shellCmds{
		// parseCmd(`mkdir -p local`),
		// parseCmd(`curl -sLOJ https://dl.google.com/go/go1.14.3.linux-amd64.tar.gz`).ChDir(`local`),
		// parseCmd(`tar -xvf go1.14.3.linux-amd64.tar.gz`).ChDir(`local`),

		parseCmd(`rm -fr .kungfu`),
		parseCmd(`mkdir -p .kungfu`),
		parseCmd(`ls .kungfu`),
		parseCmd(`git clone https://github.com/lsds/KungFu`).ChDir(`.kungfu`),
		// parseCmd(`git checkout master`).ChDir(`.kungfu/KungFu`),
		parseCmd(`go install -v ./...`).ChDir(`.kungfu/KungFu`).Env(`PATH`, `$HOME/local/go/bin:$PATH`),
	}

	var wg sync.WaitGroup
	for _, h := range hl {
		wg.Add(1)
		go func(hostname string) {
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
	envs  job.Envs
	chdir *string
}

func (s shellCmd) Env(k, v string) shellCmd {
	if s.envs == nil {
		s.envs = make(job.Envs)
	}
	s.envs[k] = v
	return s
}

func (s shellCmd) ChDir(dir string) shellCmd {
	s.chdir = &dir
	return s
}

func parseCmd(line string) shellCmd {
	parts := strings.Split(line, " ")
	assert.True(len(parts) > 0)
	return shellCmd{prog: parts[0], args: parts[1:]}
}

type shellCmds []shellCmd

func (ss shellCmds) RunOn(hostname string) {
	for i, c := range ss {
		p := job.Proc{
			Name:     fmt.Sprintf("%s#%d", hostname, i),
			Prog:     c.prog,
			Args:     c.args,
			Hostname: hostname,
			Envs:     c.envs,
			ChDir:    c.chdir,
		}
		log.Infof("running on %s $ %s %q", p.Hostname, p.Prog, p.Args)
		if err := remote.RemoteRunAll(context.TODO(), *flg.usr, []job.Proc{p}, true, *flg.logDir); err != nil {
			log.Errorf("failed to run %s $ %s %q: %v", p.Hostname, p.Prog, p.Args, err)
		}
	}
}
