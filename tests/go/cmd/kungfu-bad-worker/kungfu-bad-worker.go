package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/peer"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	runFor     = flag.Duration("run-for", 30*time.Second, "")
	errorAfter = flag.Duration("error-after", 5*time.Second, "")
)

func main() {
	flag.Parse()
	peer, err := peer.New()
	if err != nil {
		utils.ExitErr(err)
	}
	peer.Start()
	defer peer.Close()
	rank := peer.CurrentSession().Rank()
	fmt.Printf("OK, rank=%d.\n", rank)
	fmt.Fprintf(os.Stderr, "Err, rank=%d!\n", rank)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if rank == 0 {
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
