package remote

import (
	"context"
	"fmt"
	"path"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lsds/KungFu/srcs/go/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/utils/iostream"
	"github.com/lsds/KungFu/srcs/go/utils/ssh"
	"github.com/lsds/KungFu/srcs/go/utils/xterm"
)

func RemoteRunAll(ctx context.Context, user string, ps []job.Proc, verboseLog bool, logDir string) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	var fail int32
	for i, p := range ps {
		wg.Add(1)
		go func(i int, p job.Proc) {
			defer wg.Done()
			t0 := time.Now()
			config := ssh.Config{
				Host: p.PubAddr,
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
