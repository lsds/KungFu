package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/lsds/KungFu/srcs/go/kungfu/elastic"
	"github.com/lsds/KungFu/srcs/go/kungfu/peer"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	idxFile     = flag.String("idx-file", "", "TFRecord index filename.")
	maxProgress = flag.Uint64("max-progress", 0, "max progress")
	batchSize   = flag.Int("batch-size", 1, "")
)

func main() {
	flag.Parse()
	peer, err := peer.New()
	if err != nil {
		utils.ExitErr(err)
	}
	sess := peer.CurrentSession()

	var state elastic.State
	if err := peer.RestoreProgress(&state); err != nil {
		utils.ExitErr(err)
	}
	state.Progress = peer.InitProgress()

	e := json.NewEncoder(os.Stdout)
	e.SetIndent("", "  ")
	e.Encode(state)

	fmt.Printf("%s\n", "Hello World")
	fmt.Printf("Using Index: %s\n", *idxFile)

	fmt.Printf("%s\n", peer)
	fmt.Printf("%s\n", sess)
	fmt.Printf("%s\n", state)
	run(state.Progress, *maxProgress, *batchSize)
}

func run(init uint64, fini uint64, batchSize int) {
	for i := init; i < fini; i += uint64(batchSize) {
		fmt.Printf("i=%d\n", i)
	}
}
