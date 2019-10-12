package main

import (
	"bytes"
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
		testAllReduce,
		testGetPeerLatencies,
	}
	for i, t := range tests {
		fmt.Printf("test: %d\n", i)
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

func testAllReduce(kungfu *kf.Kungfu) {
	sess := kungfu.CurrentSession()
	{
		x := kb.NewVector(1, kb.I32)
		y := kb.NewVector(1, kb.I32)
		w := kf.Workspace{SendBuf: x, RecvBuf: y, OP: kb.SUM, Name: "0"}
		sess.AllReduce(w)
	}
	{
		bs := make([]byte, 1)
		n := len(bs)
		x := &kb.Vector{Data: bs, Count: n, Type: kb.U8}
		y := kb.NewVector(n, kb.U8)
		w := kf.Workspace{SendBuf: x, RecvBuf: y, OP: kb.MAX, Name: "1"}
		sess.AllReduce(w)
	}
	{
		b := &bytes.Buffer{}
		fmt.Fprintf(b, "0")
		bs := b.Bytes() // !
		n := len(bs)
		x := &kb.Vector{Data: bs, Count: n, Type: kb.U8}
		y := kb.NewVector(n, kb.U8)
		w := kf.Workspace{SendBuf: x, RecvBuf: y, OP: kb.MAX, Name: "2"}
		sess.AllReduce(w) // panic: runtime error: cgo argument has Go pointer to Go pointer
	}
}
