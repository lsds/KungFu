package remote

import (
	"context"
	"errors"
	"fmt"
	"io/ioutil"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lsds/KungFu/srcs/go/iostream"
	"github.com/lsds/KungFu/srcs/go/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/ssh"
	"github.com/lsds/KungFu/srcs/go/xterm"
)

func RemoteRunAll(ctx context.Context, user string, ps []job.Proc, verboseLog bool) ([]*Outputs, error) {
	outputs := make([]*Outputs, len(ps))
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
				log.Errorf("%s #<%s> failed to new SSH Client with config: %v: %v", xterm.Red.S("[E]"), p.Name, config, err)
				atomic.AddInt32(&fail, 1)
				outputs[i] = &Outputs{}
				return
			}
			outWatcher := iostream.NewStreamWatcher(fmt.Sprintf("%s::stdout", p.Name), verboseLog)
			errWatcher := iostream.NewStreamWatcher(fmt.Sprintf("%s::stderr", p.Name), verboseLog)
			getOutputs := func() *Outputs {
				return &Outputs{
					Stdout: outWatcher.Wait(),
					Stderr: errWatcher.Wait(),
				}
			}
			if err := client.Watch(ctx, p.Script(), outWatcher.Watch, errWatcher.Watch); err != nil {
				log.Errorf("%s #<%s> exited with error: %v, took %s", xterm.Red.S("[E]"), p.Name, err, time.Since(t0))
				atomic.AddInt32(&fail, 1)
				outputs[i] = getOutputs()
				return
			}
			outputs[i] = getOutputs()
			log.Infof("%s #<%s> finished successfully, took %s", xterm.Green.S("[I]"), p.Name, time.Since(t0))
		}(i, p)
	}
	wg.Wait()
	if fail != 0 {
		return outputs, fmt.Errorf("%d peers failed", fail)
	}
	return outputs, nil
}

// Outputs stores stdout/stderr of a process
type Outputs struct {
	Stdout []string
	Stderr []string
}

func (r *Outputs) SaveTo(prefix string) error {
	var errs []error
	if r.Stdout != nil {
		if err := ioutil.WriteFile(prefix+".stdout.log", []byte(strings.Join(r.Stdout, "\n")), 0666); err != nil {
			errs = append(errs, err)
		}
	}
	if r.Stderr != nil {
		if err := ioutil.WriteFile(prefix+".stderr.log", []byte(strings.Join(r.Stderr, "\n")), 0666); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return errors.New("failed to save some files")
	}
	return nil
}
