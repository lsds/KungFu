package runner

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"

	"github.com/luomai/kungfu/srcs/go/iostream"
	sch "github.com/luomai/kungfu/srcs/go/scheduler"
	"github.com/luomai/kungfu/srcs/go/ssh"
	"github.com/luomai/kungfu/srcs/go/xterm"
)

func RemoteRunAll(ctx context.Context, user string, ps []sch.Proc, verboseLog bool) ([]*Result, error) {
	results := make([]*Result, len(ps))
	var wg sync.WaitGroup
	var fail int32
	for i, p := range ps {
		wg.Add(1)
		go func(i int, p sch.Proc) {
			defer wg.Done()
			config := ssh.Config{
				Host: p.PubAddr,
				User: user,
			}
			client, err := ssh.New(config)
			if err != nil {
				log.Printf("%s #%s failed to new SSH Client with config: %v: %v", xterm.Red.S("[E]"), p.Name, config, err)
				atomic.AddInt32(&fail, 1)
				results[i] = &Result{}
				return
			}
			outWatcher := iostream.NewStreamWatcher(fmt.Sprintf("%s::stdout", p.Name), verboseLog)
			errWatcher := iostream.NewStreamWatcher(fmt.Sprintf("%s::stderr", p.Name), verboseLog)
			genResult := func() *Result {
				return &Result{
					Stdout: outWatcher.Wait(),
					Stderr: errWatcher.Wait(),
				}
			}
			if err := client.Watch(ctx, p.Script(), outWatcher.Watch, errWatcher.Watch); err != nil {
				log.Printf("%s #%s exited with error: %v", xterm.Red.S("[E]"), p.Name, err)
				atomic.AddInt32(&fail, 1)
				results[i] = genResult()
				return
			}
			results[i] = genResult()
			log.Printf("%s #%s finished successfully", xterm.Green.S("[I]"), p.Name)
		}(i, p)
	}
	wg.Wait()
	if fail != 0 {
		return results, fmt.Errorf("%d tasks failed", fail)
	}
	return results, nil
}

type Result struct {
	Stdout []string
	Stderr []string
}
