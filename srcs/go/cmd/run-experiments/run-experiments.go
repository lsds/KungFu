package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path"
	"sort"
	"sync"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	hostList   = flag.String("H", "", "comma separated list of <hostname>:<nslots>[,<public addr>]")
	user       = flag.String("u", "", "user name for ssh")
	timeout    = flag.Duration("timeout", 90*time.Second, "timeout")
	verboseLog = flag.Bool("v", true, "show task log")
	logDir = flag.String("logdir", "logs", "root directory for logs")
	logSubDir  = flag.String("l", "log_subdir", "root directory for logs")
	logName    = flag.String("n", "unnamed_log_subdir", "subdirectory name for experiment logs")
	reportFile = flag.String("r", "experiment-results.txt", "filename for report")
)

func main() {
	flag.Parse()
	restArgs := flag.Args()
	if len(restArgs) < 1 {
		utils.ExitErr(errors.New("missing program name"))
	}
	prog := restArgs[0]
	args := restArgs[1:]

	hostSpecs, err := plan.ParseHostSpec(*hostList)
	if err != nil {
		utils.ExitErr(err)
	}
	log.Printf("using %d VMs: %s", len(hostSpecs), humanizeHostSpecs(hostSpecs))


	logSubDirName := fmt.Sprintf("%s-%d", *logSubDir, time.Now().Unix())
	if *logSubDir != "log_subdir" {
		logSubDirName = *logSubDir
	}
	logDirForThisRound := path.Join(*logDir, logSubDirName)
	records, failed := runAllExperiments(logDirForThisRound, hostSpecs, prog, args, *timeout)

	writeReport(records, failed, os.Stdout)
	f, err := os.Create(*reportFile)
	if err != nil {
		log.Printf("failed to create report file: %v", err)
		return
	}
	defer f.Close()
	writeReport(records, failed, f)
}

func runAllExperiments(logDir string, hosts []plan.HostSpec, prog string, args []string, timeout time.Duration) ([]Record, int) {
	pool := make(chan plan.HostSpec, len(hosts))
	for _, h := range hosts {
		pool <- h
	}
	var banker sync.Mutex
	requireN := func(n int) []plan.HostSpec {
		tk := time.NewTicker(1 * time.Second)
		defer tk.Stop()
		for {
			got := func() []plan.HostSpec {
				banker.Lock()
				banker.Unlock()
				if len(pool) >= n {
					var hs []plan.HostSpec
					for i := 0; i < n; i++ {
						hs = append(hs, <-pool)
					}
					return hs
				}
				return nil
			}()
			if got != nil {
				return got
			}
			<-tk.C
		}
	}
	returnAll := func(hs []plan.HostSpec) {
		for _, h := range hs {
			pool <- h
		}
	}

	var wg sync.WaitGroup
	var records []Record
	var lock sync.Mutex
	var lastID int
	run := func(algo kb.KungFu_AllReduceAlgo, partition []int) {
		if len(hosts) < len(partition) {
			return // total resource not sufficient
		}
		wg.Add(1)
		lastID++
		go func(id int) {
			defer wg.Done()
			hs := requireN(len(partition))
			defer func() { returnAll(hs) }()
			log.Printf("begin experiment {%s %v} on {%s}", algo, partition, humanizeHostSpecs(hs))
			t0 := time.Now()
			myLogDir := path.Join(logDir, fmt.Sprintf("%d", id))
			res, err := runExperiment(myLogDir, hs, prog, args, algo, partition, timeout)
			if err != nil {
				log.Printf("failed experiment {%s %v} with: %v", algo, partition, err)
				return
			}
			r := Record{
				ID:        id,
				Took:      time.Since(t0),
				Algo:      algo,
				Partition: partition,
				Result:    *res,
			}
			log.Printf("end experiment {%s %v} on {%s} with: %s", algo, partition, humanizeHostSpecs(hs), r)
			lock.Lock()
			records = append(records, r)
			log.Printf("experiment #%d finished, %d experiments finished so far", id, len(records))
			lock.Unlock()
		}(lastID)
	}

	algos := []kb.KungFu_AllReduceAlgo{
		//kb.KungFu_Star,
		kb.KungFu_Ring,
		//kb.KungFu_Clique,
		//kb.KungFu_Tree,
	}
	for _, a := range algos {
		// run(a, []int{1})
		// run(a, []int{2})
		// run(a, []int{3})
		 run(a, []int{4})

		// run(a, []int{1, 3})
		// run(a, []int{2, 2})
		// run(a, []int{3, 3})
		// run(a, []int{4, 4})
		//run(a, []int{1, 1, 1, 1})
	}

	wg.Wait()
	sort.Slice(records, func(i, j int) bool { return records[i].ID < records[j].ID })
	return records, lastID - len(records)
}
