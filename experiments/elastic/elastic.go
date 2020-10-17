package elastic

import (
	"path"
	"strconv"

	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/job"
	"github.com/lsds/KungFu/srcs/go/plan"
)

const script = `benchmarks/scaling/benchmark_kungfu_scaling.py`

var str = strconv.Itoa

type Experiment struct {
	BatchSize  int
	TrainSteps int
	Epochs     int
	EpochSize  int
	Schedule   string
}

func Default() Experiment {
	return Experiment{
		BatchSize:  1,
		TrainSteps: 100,
		Epochs:     1,
		EpochSize:  1000,
		Schedule:   `10:1,100:0`,
	}
}

func (e Experiment) Job(id string, kfRoot string, configServer string, strategy base.Strategy, hl plan.HostList, pr plan.PortRange, logDir string) job.Job {
	const prog = `python3`
	args := []string{
		path.Join(kfRoot, script),

		`--model-dir`, `.kungfu/ckpt/` + id,

		`--batch-size`, str(e.BatchSize),
		`--train-steps`, str(e.TrainSteps),

		`--epochs`, str(e.Epochs),
		`--epoch-size`, str(e.EpochSize),

		`--tf-method`, `estimator`,
		`--elastic`,
		`--resize-schedule`, e.Schedule,

		`--show-training-throughput`,
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
