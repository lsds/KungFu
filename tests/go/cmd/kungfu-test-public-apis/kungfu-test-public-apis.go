package main

import (
	"bytes"
	"flag"
	"fmt"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func main() {
	flag.Parse()
	kungfu, err := kf.New()
	if err != nil {
		utils.ExitErr(err)
	}
	kungfu.Start()
	defer kungfu.Close()
	tests := []func(*kf.Kungfu){
		// TODO: more tests
		testAllReduce,
		testGetPeerLatencies,
		testP2P,
	}
	for i, t := range tests {
		fmt.Printf("# test: %d\n", i)
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
	sess.Barrier()
	fmt.Printf("%s OK\n", `testGetPeerLatencies`)
}

func testAllReduce(kungfu *kf.Kungfu) {
	const step = 20
	sess := kungfu.CurrentSession()
	np := sess.ClusterSize()
	for i := 0; i < step; i++ {
		fmt.Printf("step: %d\n", i)
		{
			x := kb.NewVector(1, kb.I32)
			y := kb.NewVector(1, kb.I32)
			z := kb.NewVector(1, kb.I32)
			x.AsI32()[0] = 1
			z.AsI32()[0] = int32(np)
			w := kf.Workspace{SendBuf: x, RecvBuf: y, OP: kb.SUM, Name: "0"}
			sess.AllReduce(w)
			if !utils.BytesEq(y.Data, z.Data) {
				utils.ExitErr(fmt.Errorf("%s failed", `testAllReduce`))
			}
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
			bs := b.Bytes()
			n := len(bs)
			x := &kb.Vector{Data: bs, Count: n, Type: kb.U8}
			y := kb.NewVector(n, kb.U8)
			w := kf.Workspace{SendBuf: x, RecvBuf: y, OP: kb.MAX, Name: "2"}
			sess.AllReduce(w)
		}
	}
	fmt.Printf("%s OK\n", `testAllReduce`)
}

func testP2P(kungfu *kf.Kungfu) {
	const step = 20
	const count = 10
	sess := kungfu.CurrentSession()
	np := sess.ClusterSize()
	rank := sess.Rank()
	name := "weight"

	a := kb.NewVector(count, kb.I32)
	b := kb.NewVector(count, kb.I32)
	c := kb.NewVector(count, kb.I32)
	x := a.AsI32()
	z := c.AsI32()

	for i := 0; i < step; i++ {
		target := (rank + 1) % np
		fmt.Printf("step=%d, rank=%d, target=%d\n", i, rank, target)
		for j := 0; j < count; j++ {
			x[j] = int32(i * rank)
			z[j] = int32(i * target)
		}
		if err := sess.Barrier(); err != nil {
			utils.ExitErr(err)
		}
		if err := kungfu.Save(name, a); err != nil {
			utils.ExitErr(err)
		}
		if err := sess.Barrier(); err != nil {
			utils.ExitErr(err)
		}
		if ok, err := sess.Request(target, "", name, b); !ok || err != nil {
			utils.ExitErr(fmt.Errorf("%s failed", `testP2P`))
		}
		if !utils.BytesEq(b.Data, c.Data) {
			utils.ExitErr(fmt.Errorf("%s failed", `testP2P`))
		}
	}
	if err := sess.Barrier(); err != nil {
		utils.ExitErr(err)
	}
	for i := 0; i < step; i++ {
		target := (rank + 1) % np
		fmt.Printf("step=%d, rank=%d, target=%d, should fail\n", i, rank, target)
		if ok, err := sess.Request(target, "", name+"!", b); ok || err != nil {
			utils.ExitErr(fmt.Errorf("%s failed", `testP2P`))
		}
	}
	if err := sess.Barrier(); err != nil {
		utils.ExitErr(err)
	}
	fmt.Printf("%s OK\n", `testP2P`)
}
