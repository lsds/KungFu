package runner

import (
	"context"
	"github.com/lsds/KungFu/srcs/go/kungfu/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/runner/local"
	"strconv"
	"strings"
)

func SimpleRun(ctx context.Context, selfIPv4 uint32, cluster plan.Cluster, j job.Job, verboseLog bool, Monitor int) {
	procs := j.CreateProcs(cluster, selfIPv4)
	log.Infof("will parallel run %d instances of %s with %q", len(procs), j.Prog, j.Args)
	d, err := utils.Measure(func() error { return local.RunAll(ctx, procs, verboseLog, Monitor) })
	log.Infof("all %d/%d local peers finished, took %s", len(procs), len(cluster.Workers), d)
	if err != nil {
		erros := err.Error()
		datas := strings.Split(erros, ":")
		if datas[0] == "server dump" {
			for key, value := range j.Args {
				if value == "--n-epochs" || value == "--num-epochs" {
					epochfi, err := strconv.Atoi(datas[1])
					if err != nil {
					}
					epochini, err := strconv.Atoi(j.Args[key+1])
					if err != nil {
					}
					j.Args[key+1] = strconv.Itoa(epochini - epochfi)
				}
			}
			j.Args = append(j.Args, "--restart")
			j.Args = append(j.Args, "1")
			SimpleRun(ctx, selfIPv4, cluster, j, verboseLog, Monitor)
		} else {
			utils.ExitErr(err)
		}
	}
}
