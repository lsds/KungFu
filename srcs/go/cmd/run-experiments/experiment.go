package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/runner"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func writeReport(records []Record, failed int, f io.Writer) {
	fmt.Fprintf(f, "Got results from %d experiments, %d failed\n", len(records), failed)
	for i, r := range records {
		fmt.Fprintf(f, "#%d %s\n", i, r)
	}
}

type Record struct {
	ID        int
	Took      time.Duration
	Partition []int
	Algo      kb.KungFu_AllReduceAlgo
	Result    Result
}

func (r Record) String() string {
	return fmt.Sprintf("%s %v %s took %s", r.Algo, r.Partition, r.Result, r.Took)
}

type Result struct {
	Mean float32
	Conf float32
}

func (r Result) String() string {
	return fmt.Sprintf("%f +-%f", r.Mean, r.Conf)
}

func parseResult(line string, r *Result) {
	fmt.Sscanf(line, `Img/sec per /gpu:0: %f +-%f`, &r.Mean, &r.Conf)
}

func fmtHostSpecs(hosts []plan.HostSpec) string {
	var ss []string
	for _, h := range hosts {
		ss = append(ss, h.String())
	}
	return strings.Join(ss, ",")
}

func humanizeHostSpecs(hosts []plan.HostSpec) string {
	var ss []string
	for _, h := range hosts {
		ss = append(ss, fmt.Sprintf("<ip=%s, slots=%d, pub_ip=%s>", h.Hostname, h.Slots, h.PublicAddr))
	}
	return strings.Join(ss, ", ")
}

func grep(pattern string, input []string) []string {
	var lines []string
	for _, line := range input {
		if strings.Contains(line, pattern) {
			lines = append(lines, line)
		}
	}
	return lines
}

func runExperiment(logDir string, hosts []plan.HostSpec, prog string, args []string, algo kb.KungFu_AllReduceAlgo, partition []int, timeout time.Duration) (*Result, error) {
	if err := os.MkdirAll(logDir, os.ModePerm); err != nil {
		return nil, err
	}
	lf, err := os.Create(path.Join(logDir, "readme.txt"))
	if err != nil {
		return nil, err
	}
	defer lf.Close()
	fmt.Fprintf(lf, "%s\n", humanizeHostSpecs(hosts))
	fmt.Fprintf(lf, "%s %v\n", algo.String(), partition)

	hosts, err = reschedule(hosts, partition)
	if err != nil {
		return nil, err
	}

	jc := sch.JobConfig{
		TaskCount: plan.TotalCap(hosts),
		HostList:  fmtHostSpecs(hosts),
		Prog:      prog,
		Args:      args,
	}
	ps, err := jc.CreateProcs(algo)
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var res Result
	d, err := utils.Measure(func() error {
		outputs, err := runner.RemoteRunAll(ctx, *user, ps, *verboseLog)
		for i, o := range outputs {
			o.SaveTo(path.Join(logDir, fmt.Sprintf("task-%02d", i)))
		}
		for _, o := range outputs {
			if info := grep(`Img/sec per /gpu:0`, o.Stdout); len(info) > 0 {
				parseResult(info[0], &res)
				break
			}
		}
		return err
	})
	log.Printf("all %d tasks finished, took %s", len(ps), d)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(lf, "%s\n", res)
	return &res, nil
}

func reschedule(hosts []plan.HostSpec, partition []int) ([]plan.HostSpec, error) {
	if len(hosts) < len(partition) {
		return nil, errors.New("hosts not enough")
	}
	var workers []plan.HostSpec
	for i, p := range partition {
		w := hosts[i]
		if w.Slots < p {
			return nil, errors.New("host slots not enough")
		}
		w.Slots = p
		workers = append(workers, w)
	}
	return workers, nil
}
