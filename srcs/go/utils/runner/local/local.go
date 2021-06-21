package local

import (
	"context"
	"fmt"
	"os/exec"
	"path"
	"strings"
	"sync"
	"sync/atomic"
    "strconv"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/proc"
	"github.com/lsds/KungFu/srcs/go/utils/iostream"
	"github.com/lsds/KungFu/srcs/go/utils/xterm"
)

type Runner struct {
	Name          string
	Color         xterm.Color
	LogDir        string
	LogFilePrefix string
	VerboseLog    bool
}

// Run a command with context
func (r Runner) Run(cmd *exec.Cmd) error {
	return runWith(r.defaultRedirectors(), cmd)
}

func (r Runner) defaultRedirectors() []*iostream.StdWriters {
	var redirectors []*iostream.StdWriters
	if r.VerboseLog {
		redirectors = append(redirectors, iostream.NewXTermRedirector(r.Name, r.Color))
	}
	if len(r.LogFilePrefix) > 0 {
		redirectors = append(redirectors, iostream.NewFileRedirector(path.Join(r.LogDir, r.LogFilePrefix)))
	}
	return redirectors
}

func runWith(redirectors []*iostream.StdWriters, cmd *exec.Cmd) error {
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	defer stdout.Close()
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	defer stderr.Close()
	results := iostream.StdReaders{Stdout: stdout, Stderr: stderr, Flagdown: 0, Epochfinish: 0}
	ioDone := results.Stream(redirectors...)
	if err := cmd.Start(); err != nil {
		return err
	}
	ioDone.Wait() // call this before cmd.Wait!
	errs := cmd.Wait()
	if errs == nil {
		if results.Flagdown != 0{
			return fmt.Errorf("some machine died:"+strconv.Itoa(results.Epochfinish))
		} else{
			return errs
		}
	}else{
		return errs
	}
}

func RunAll(ctx context.Context, ps []proc.Proc, verboseLog bool, Monitor int) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	var fail int32
	var dumpmachine int32
    epochsfi := ""
    if Monitor == 1 {
        wg.Add(1)
        go func(){
            machines := "-machines="+strconv.Itoa(len(ps))
            args := []string{"run","srcs/go/cmd/kungfu-recovery/monitor.go", machines}
            r := &Runner{
            Name:          "monitor",
            Color:         xterm.BasicColors.Choose(0),
            VerboseLog:    verboseLog,
            LogFilePrefix: strings.Replace("monitor", "/", "-", -1),
            LogDir:        "",
        }
        if err := r.Run(exec.Command("go", args...)); err != nil {
            log.Errorf("exited with error: %v", err)
            errorst := err.Error()
            datas := strings.Split(errorst, ":")
            if datas[0] == "some machine died"{
                atomic.AddInt32(&dumpmachine, 1)
                epochsfi = datas[1]
            }
            atomic.AddInt32(&fail, 1)
            cancel()
        } else {
            log.Infof("finished successfully")
        }
        wg.Done()
        }()
    }

	for i, p := range ps {
		wg.Add(1)
		go func(i int, p proc.Proc) {
			r := &Runner{
				Name:          p.Name,
				Color:         xterm.BasicColors.Choose(i),
				VerboseLog:    verboseLog,
				LogFilePrefix: strings.Replace(p.Name, "/", "-", -1),
				LogDir:        p.LogDir,
			}
			if err := r.TryRun(ctx, p); err != nil {
				log.Errorf("#<%s> exited with error: %v", p.Name, err)
				atomic.AddInt32(&fail, 1)
				cancel()
			} else {
				log.Debugf("#<%s> finished successfully", p.Name)
			}
			wg.Done()
		}(i, p)
	}
	wg.Wait()
	if fail != 0 {
		if dumpmachine != 0{
			return fmt.Errorf("server dump:"+epochsfi)
		}else{
			return fmt.Errorf("%d tasks failed", fail)
		}
	}
	return nil
}
