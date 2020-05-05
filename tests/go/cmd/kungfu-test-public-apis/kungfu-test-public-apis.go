package main

import (
	"bytes"
	"flag"
	"fmt"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/peer"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func main() {
	flag.Parse()
	p, err := peer.New()
	if err != nil {
		utils.ExitErr(err)
	}
	p.Start()
	defer p.Close()
	tests := []func(*peer.Peer){
		// TODO: more tests
		testAllReduce,
		testAllReduceWith,
		testAllGather,
		testGetPeerLatencies,
		testP2P,
	}
	for i, t := range tests {
		fmt.Printf("# test: %d\n", i)
		t(p)
	}
	fmt.Printf("All Tests OK\n")
}

func testGetPeerLatencies(peer *peer.Peer) {
	sess := peer.CurrentSession()
	latencies := sess.GetPeerLatencies()
	fmt.Printf("rank: %d\n", sess.Rank())
	for _, d := range latencies {
		fmt.Printf("%12s", d)
	}
	fmt.Printf("\n")
	assert.OK(sess.Barrier())
	fmt.Printf("%s OK\n", `testGetPeerLatencies`)
}

func testAllReduce(peer *peer.Peer) {
	const step = 20
	sess := peer.CurrentSession()
	np := sess.Size()
	for i := 0; i < step; i++ {
		fmt.Printf("step: %d\n", i)
		{
			x := kb.NewVector(1, kb.I32)
			y := kb.NewVector(1, kb.I32)
			z := kb.NewVector(1, kb.I32)
			x.AsI32()[0] = 1
			z.AsI32()[0] = int32(np)
			w := kb.Workspace{SendBuf: x, RecvBuf: y, OP: kb.SUM, Name: "0"}
			assert.OK(sess.AllReduce(w))
			assert.True(utils.BytesEq(y.Data, z.Data))
		}
		{
			bs := make([]byte, 1)
			n := len(bs)
			x := &kb.Vector{Data: bs, Count: n, Type: kb.U8}
			y := kb.NewVector(n, kb.U8)
			w := kb.Workspace{SendBuf: x, RecvBuf: y, OP: kb.MAX, Name: "1"}
			assert.OK(sess.AllReduce(w))
		}
		{
			b := &bytes.Buffer{}
			fmt.Fprintf(b, "0")
			bs := b.Bytes()
			n := len(bs)
			x := &kb.Vector{Data: bs, Count: n, Type: kb.U8}
			y := kb.NewVector(n, kb.U8)
			w := kb.Workspace{SendBuf: x, RecvBuf: y, OP: kb.MAX, Name: "2"}
			assert.OK(sess.AllReduce(w))
		}
	}
	fmt.Printf("%s OK\n", `testAllReduce`)
}

func testAllReduceWith(peer *peer.Peer) {
	sess := peer.CurrentSession()
	np := sess.Size()
	tree := make([]int32, np)
	var root int32
	for i := 0; i < np; i++ {
		tree[i] = root
	}
	size := 1 << 20
	x := kb.NewVector(size, kb.I32)
	y := kb.NewVector(size, kb.I32)
	z := kb.NewVector(size, kb.I32)
	fillI32(x.AsI32(), 1)
	fillI32(z.AsI32(), int32(np))
	w := kb.Workspace{SendBuf: x, RecvBuf: y, OP: kb.SUM, Name: "0"}
	assert.OK(sess.AllReduceWith(tree, w))
	assert.True(utils.BytesEq(y.Data, z.Data))
}

func testAllGather(peer *peer.Peer) {
	sess := peer.CurrentSession()
	np := sess.Size()
	rank := sess.Rank()
	count := 1024
	w := kb.Workspace{
		SendBuf: kb.NewVector(count, kb.I32),
		RecvBuf: kb.NewVector(count*np, kb.I32),
		Name:    "0",
	}
	x := w.SendBuf.AsI32()
	y := w.RecvBuf.AsI32()
	for i := range x {
		x[i] = int32(rank + 1)
	}
	ySum := int32(np * (np + 1) / 2 * count)
	step := 10
	for i := 0; i < step; i++ {
		assert.OK(sess.AllGather(w))
		if s := sumI32(y); s != ySum {
			utils.ExitErr(fmt.Errorf("%s failed", "testAllGather"))
		}
	}
	fmt.Printf("%s OK\n", `testAllGather`)
}

func testP2P(peer *peer.Peer) {
	const step = 20
	const count = 10
	sess := peer.CurrentSession()
	np := sess.Size()
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
		assert.OK(sess.Barrier())
		if err := peer.Save(name, a); err != nil {
			utils.ExitErr(err)
		}
		assert.OK(sess.Barrier())
		if ok, err := peer.RequestRank(target, "", name, b); !ok || err != nil {
			utils.ExitErr(fmt.Errorf("%s failed", `testP2P`))
		}
		assert.True(utils.BytesEq(b.Data, c.Data))
	}
	assert.OK(sess.Barrier())
	for i := 0; i < step; i++ {
		target := (rank + 1) % np
		fmt.Printf("step=%d, rank=%d, target=%d, should fail\n", i, rank, target)
		if ok, err := peer.RequestRank(target, "", name+"!", b); ok || err != nil {
			utils.ExitErr(fmt.Errorf("%s failed", `testP2P`))
		}
	}
	assert.OK(sess.Barrier())
	fmt.Printf("%s OK\n", `testP2P`)
}

func sumI32(xs []int32) int32 {
	var s int32
	for _, x := range xs {
		s += x
	}
	return s
}

func fillI32(xs []int32, x int32) {
	for i := range xs {
		xs[i] = x
	}
}
