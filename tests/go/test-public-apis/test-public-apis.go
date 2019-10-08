package main

import (
	"flag"
	"fmt"
	"os"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func main() {
	flag.Parse()
	config := kf.Config{
		Strategy: kb.ParseStrategy(os.Getenv(kb.AllReduceStrategyEnvKey)),
	}
	kungfu, err := kf.New(config)
	if err != nil {
		utils.ExitErr(err)
	}
	kungfu.Start()
	defer kungfu.Close()

	tests := []func(*kf.Kungfu){
		// TODO: more tests
		testGetPeerLatencies,
	}
	for _, t := range tests {
		t(kungfu)
	}
}

func testGetPeerLatencies(kungfu *kf.Kungfu) {
	sess := kungfu.CurrentSession()
	latencies := sess.GetPeerLatencies()
	fmt.Printf("rank: %d\n", sess.Rank())
	for _, d := range latencies {
		fmt.Printf("%12s", d)
	}
	fmt.Printf("\n")
}
