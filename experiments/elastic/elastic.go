package elastic

import (
	"path"
	"strconv"

	"github.com/lsds/KungFu/srcs/go/job"
	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
)

const prefix = `./code/repos/github.com/lsds`

const kfRoot = `./KungFu`

const script = `benchmarks/scaling/benchmark_kungfu_scaling.py`

var str = strconv.Itoa

func TestJob(id string, configServer string, strategy base.Strategy, hl plan.HostList, pr plan.PortRange, logDir string) job.Job {
	const prog = `python3`
	args := []string{
		path.Join(prefix, kfRoot, script),

		`--model-dir`, `.kungfu/ckpt/` + id,

		`--batch-size`, str(1),
		`--train-steps`, str(100),

		`--epochs`, str(1),
		`--epoch-size`, str(100),

		`--tf-method`, `estimator`,
		`--elastic`,
	}

	return job.Job{
		ConfigServer: configServer,
		Strategy:     strategy,
		HostList:     hl,
		Prog:         prog,
		Args:         args,
		LogDir:       logDir,
	}
}
