package tfkeras

import (
	"fmt"
	"path"

	"github.com/lsds/KungFu/srcs/go/job"
	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type Model string

const (
	MobileNetV2 Model = `MobileNetV2`
	ResNet50    Model = `ResNet50`
)

var Models = []Model{
	MobileNetV2,
	ResNet50,
}

type KFOptimizer string

const (
	SyncSgd     KFOptimizer = `sync-sgd`
	SyncSgdNccl KFOptimizer = `sync-sgd-nccl`
)

var Optimizers = []KFOptimizer{
	SyncSgd,
	SyncSgdNccl,
}

type Experiment struct {
	Model     Model
	BatchSize int

	WarmupBatches   int
	NumIters        int
	NumBatchPerIter int

	KFOptimizer KFOptimizer
}

const script = `benchmarks/system/benchmark_kungfu.py`

func (e Experiment) Job(strategy base.Strategy, hl plan.HostList, pr plan.PortRange, logDir string) job.Job {
	prog := `python3`
	args := []string{
		path.Join(`./KungFu`, script),
		fmt.Sprintf("--batch-size %d", e.BatchSize),
		fmt.Sprintf("--num-warmup-batches %d", e.WarmupBatches),
		fmt.Sprintf("--num-iters %d", e.NumIters),
		fmt.Sprintf("--num-batches-per-iter %d", e.NumBatchPerIter),
		fmt.Sprintf("--kf-opt %s", e.KFOptimizer),
	}

	return job.Job{
		Strategy:  strategy,
		HostList:  hl,
		PortRange: pr,
		Prog:      prog,
		Args:      args,
		LogDir:    logDir,
	}
}

func Default() []Experiment {
	return Combination(
		[]Model{
			MobileNetV2,
			ResNet50,
		},
		[]KFOptimizer{
			SyncSgd,
			SyncSgdNccl,
		},
		[]int{32},
	)
}

func Combination(models []Model, optimizers []KFOptimizer, batchSizes []int) []Experiment {
	var es []Experiment
	for _, m := range models {
		for _, opt := range optimizers {
			for _, bs := range batchSizes {
				e := Experiment{
					Model:           m,
					BatchSize:       bs,
					WarmupBatches:   4,
					NumIters:        4,
					NumBatchPerIter: 4,
					KFOptimizer:     opt,
				}
				es = append(es, e)
			}
		}
	}
	return es
}
