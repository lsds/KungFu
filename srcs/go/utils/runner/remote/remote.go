package remote

import (
	"context"
	"fmt"
	"path"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/job"
	"github.com/lsds/KungFu/srcs/go/kungfu/runtime"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/proc"
	"github.com/lsds/KungFu/srcs/go/utils/iostream"
	"github.com/lsds/KungFu/srcs/go/utils/ssh"
	"github.com/lsds/KungFu/srcs/go/utils/xterm"
)

func RemoteRunAll(ctx context.Context, user string, ps []proc.Proc, verboseLog bool, logDir string) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	var fail int32
	for i, p := range ps {
		wg.Add(1)
		go func(i int, p proc.Proc) {
			defer wg.Done()
			t0 := time.Now()
			config := ssh.Config{
				Host: p.Hostname,
				User: user,
			}
			client, err := ssh.New(config)
			if err != nil {
				log.Errorf("#<%s> failed to new SSH Client with config: %v: %v", p.Name, config, err)
				atomic.AddInt32(&fail, 1)
				return
			}
			var redirectors []*iostream.StdWriters
			if verboseLog {
				redirectors = append(redirectors, iostream.NewXTermRedirector(p.Name, xterm.BasicColors.Choose(i)))
			}
			redirectors = append(redirectors, iostream.NewFileRedirector(path.Join(logDir, p.Name)))
			if err := client.Watch(ctx, p.Script(), redirectors); err != nil {
				log.Errorf("#<%s> exited with error: %v, took %s", p.Name, err, time.Since(t0))
				atomic.AddInt32(&fail, 1)
				cancel()
				return
			}
			log.Debugf("#<%s> finished successfully, took %s", p.Name, time.Since(t0))
		}(i, p)
	}
	wg.Wait()
	if fail != 0 {
		return fmt.Errorf("%d tasks failed", fail)
	}
	return nil
}

const runnerProg = `kungfu-run`

func RunStaticKungFuJob(ctx context.Context, j job.Job, sp runtime.SystemParameters, quiet bool) error {
	hl := sp.HostList
	runners := hl.GenRunnerList(sp.RunnerPort)
	runnerFlags := []string{
		`PATH=$HOME/local/python/bin:$HOME/go/bin:$PATH`, // FIXME: find kungfu-run PATH

		`PYTHONWARNINGS=ignore`,
		`TF_CPP_MIN_LOG_LEVEL=2`,
		runnerProg,
		`-np`, strconv.Itoa(sp.ClusterSize),
		`-H`, hl.String(),
		`-port-range`, sp.WorkerPortRange.String(),
		`-nic`, sp.Nic,
		`-strategy`, j.Strategy.String(),
		`-logdir`, j.LogDir,
	}
	if quiet {
		runnerFlags = append(runnerFlags, `-q`)
	}
	var ps []proc.Proc
	for _, r := range runners {
		p := proc.Proc{
			Name:     plan.FormatIPv4(r.IPv4),
			Prog:     `env`,
			Args:     append(runnerFlags, j.ProgAndArgs()...),
			Hostname: hl.LookupHost(r.IPv4),
		}
		ps = append(ps, p)
	}
	return RemoteRunAll(ctx, sp.User, ps, true, j.LogDir)
}

func RunElasticKungFuJob(ctx context.Context, j job.Job, sp runtime.SystemParameters, quiet bool) error {
	hl := sp.HostList
	runners := hl.GenRunnerList(sp.RunnerPort)
	runnerFlags := []string{
		`PATH=$HOME/local/python/bin:$HOME/go/bin:$PATH`, // FIXME: find kungfu-run PATH

		`PYTHONWARNINGS=ignore`,
		`TF_CPP_MIN_LOG_LEVEL=2`,

		runnerProg,
		`-w`,
		`-k`,
		`-config-server`, j.ConfigServer,
		`-np`, strconv.Itoa(sp.ClusterSize),
		`-H`, hl.String(),
		`-port-range`, sp.WorkerPortRange.String(),
		`-nic`, sp.Nic,
		`-strategy`, j.Strategy.String(),
		`-logdir`, j.LogDir,
	}
	if quiet {
		runnerFlags = append(runnerFlags, `-q`)
	}
	var ps []proc.Proc
	for _, r := range runners {
		p := proc.Proc{
			Name:     plan.FormatIPv4(r.IPv4),
			Prog:     `env`,
			Args:     append(runnerFlags, j.ProgAndArgs()...),
			Hostname: hl.LookupHost(r.IPv4),
		}
		ps = append(ps, p)
	}
	return RemoteRunAll(ctx, sp.User, ps, true, j.LogDir)
}
