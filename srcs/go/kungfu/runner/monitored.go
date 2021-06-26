package runner

import (
	"context"
	"github.com/lsds/KungFu/srcs/go/kungfu/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/runner/local"
	"strconv"
	"sync"
	"sync/atomic"
)

func MonitoredRun(ctx context.Context, selfIPv4 uint32, cluster plan.Cluster, j job.Job, verboseLog bool) {
	for {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		var sucessfi int32
		var cont int32
		procs := j.CreateProcs(cluster, selfIPv4)
		s := New(len(procs))
		s.wg.Add(1)
		log.Infof("will parallel run %d instances of %s with %q under monitor", len(procs), j.Prog, j.Args)
		go s.Start()
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			d, err := utils.Measure(func() error { return local.RunAll(ctx, procs, verboseLog) })
			log.Infof("all %d/%d local peers finished, took %s", len(procs), len(cluster.Workers), d)
			if err != nil {
				if cont == 1 {

				} else {
					utils.ExitErr(err)
				}
			} else {
				atomic.AddInt32(&sucessfi, 1)
			}
			wg.Done()
		}()

		Results := s.Wait()
		if Results.FinishFlag {
			atomic.AddInt32(&sucessfi, 1)
		}
		if Results.DownFlag {
			atomic.AddInt32(&cont, 1)
			cancel()
			log.Infof("some machine down")
			for key, value := range j.Args {
				if value == "--n-epochs" || value == "--num-epochs" {
					epochini, err := strconv.Atoi(j.Args[key+1])
					if err != nil {
					}
					j.Args[key+1] = strconv.Itoa(epochini - Results.Epochnum)
				}
			}
			j.Args = append(j.Args, "--restart")
			j.Args = append(j.Args, "1")
			continue
		}
		wg.Wait()
		if sucessfi == 2 {
			log.Infof("success finish")
			break
		}

	}
}