package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	runFor     = flag.Duration("run-for", 30*time.Second, "")
	errorAfter = flag.Duration("error-after", 5*time.Second, "")
)

func main() {
	flag.Parse()
	fmt.Printf("OK.\n")
	fmt.Fprintf(os.Stderr, "Err! \n")
	kungfu, err := kf.New()
	if err != nil {
		utils.ExitErr(err)
	}
	kungfu.Start()
	defer kungfu.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if kungfu.CurrentSession().Rank() == 0 {
		ctx, cancel = context.WithTimeout(ctx, *errorAfter)
	}
	done := time.After(*runFor)
	select {
	case <-ctx.Done():
		os.Exit(1)
	case <-done:
		return
	}
}
