package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	rch "github.com/luomai/kungfu/src/go/rchannel"
	"github.com/luomai/kungfu/src/go/xterm"
)

var (
	np      = flag.Int("np", runtime.NumCPU(), "number of tasks")
	timeout = flag.Duration("timeout", 10*time.Second, "timeout")
)
var (
	basicColors = []xterm.Color{
		xterm.Green,
		xterm.Yellow,
	}

	waiting []bool
	lock    sync.Mutex
)

func main() {
	flag.Parse()
	restArgs := flag.Args()
	prog := restArgs[0]
	args := restArgs[1:]

	specs := rch.GenCluster(*np)

	ctx := context.Background()
	var wg sync.WaitGroup
	wg.Add(*np)
	var fail int32
	done := make(chan int, *np)

	for i := 0; i < *np; i++ {
		go func(i int) {
			envs := []string{
				// FIXME: passdown more envs
				fmt.Sprintf(`%s=%s`, `PATH`, os.Getenv(`PATH`)),
				fmt.Sprintf(`%s=%s`, `HOME`, os.Getenv(`HOME`)),
				fmt.Sprintf(`%s=%s`, rch.ClusterSpecEnvKey, specs[i]),
			}
			c := basicColors[i%len(basicColors)]
			log.Printf("%s %q", prog, args)
			prefix := fmt.Sprintf("%02d ", i)
			ctx, cancel := context.WithTimeout(ctx, *timeout)
			defer cancel()
			if err := run(ctx, c.S(prefix), prog, args, envs); err != nil {
				log.Printf("%s #%d exited with error: %v", xterm.Red.S("[E]"), i, err)
				atomic.AddInt32(&fail, 1)
			} else {
				log.Printf("%s #%d finished successfully", xterm.Green.S("[I]"), i)
			}
			done <- i
			wg.Done()
		}(i)
	}
	wg.Wait()
	if fail != 0 {
		log.Printf("%d node failed\n", fail)
		os.Exit(1)
	}
}
