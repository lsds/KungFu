package main

import (
	"flag"
	"os"
	"path"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var home, _ = os.UserHomeDir()

var (
	batchSize = flag.Int("bs", 100, "")
	epochs    = flag.Int("epochs", 1, "")
	dataDir   = flag.String("data-dir", path.Join(home, "var/data/mnist"), "")
	normalize = flag.Bool("normalize", true, "")
)

func main() {
	flag.Parse()
	t0 := time.Now()
	mnist, err := LoadMnistDataSets(*dataDir, *normalize)
	if err != nil {
		utils.ExitErr(err)
	}
	log.Infof("LoadMnistDataSets took %s", time.Since(t0))
	model := createSLP(28*28, 10)
	trainAndEval(model, mnist)
}

func trainAndEval(model Model, ds *DataSets) {
	for epoch := 0; epoch < *epochs; epoch++ {
		samples, batches := trainAll(model, ds.Train)
		log.Infof("trained %d samples in %d batches", samples, batches)
		r := testAll(model, ds.Test)
		log.Infof("epoch %d, eval accuracy: %f", epoch, r.Accuracy())
	}
}

func trainAll(model Model, ds *DataSet) (int, int) {
	var lr float32
	return batchRun(*batchSize, ds, func(samples, labels *Tensor) {
		log.Infof("training with (%s, %s)", samples.Info(), labels.Info())
		gradVars := model.train(samples, labels)
		//
		lr = 0.1
		for _, gv := range gradVars {
			if gv.G != nil {
				AXPY(lr, gv.G, gv.V, gv.V)
			}
		}
	})
}

func testAll(model Model, ds *DataSet) BinaryResult {
	var r BinaryResult
	batchRun(*batchSize, ds, func(samples, labels *Tensor) {
		r.Add(model.test(samples, labels))
	})
	return r
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

func batchRun(batchSize int, ds *DataSet, f func(sample, Labels *Tensor)) (int, int) {
	var m int
	n := ds.Samples.Ldm()
	for i := 0; i < n; i += batchSize {
		j := min(i+batchSize, n)
		samples := ds.Samples.Slice(i, j)
		labels := ds.Labels.Slice(i, j)
		f(samples, labels)
		m++
	}
	return n, m
}
