package main

import (
	"flag"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	shv "github.com/lsds/KungFu/srcs/go/sharedvariable"
)

var (
	port = flag.Int("port", 9999, "port")
)

func main() {
	handler := shv.NewServer()
	addr := net.JoinHostPort("0.0.0.0", strconv.Itoa(*port))
	srv := &http.Server{
		Addr:    addr,
		Handler: handler,
	}

	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		srv.ListenAndServe()
		wg.Done()
	}()
	cli := shv.NewClient(addr)
	runTests(cli)
	srv.Close()
	wg.Wait()
}

func runTests(cli *shv.Client) {
	b := kb.NewBuffer(1, kb.KungFu_INT32)
	a := b.AsInt32()

	name := "v1"

	check := func(err error) {
		if err != nil {
			fmt.Printf("failed: %v\n", err)
		}
	}

	check(cli.Create("v1", b.Count, b.Type))
	{
		a[0] = 1
		check(cli.Put(name, b))
	}
	{
		a[0] = 0
		check(cli.Get(name, b))
		fmt.Printf("a[0] = %d\n", a[0])
	}

	for i := 0; i < 30; i++ {
		check(cli.Add(name, b))
		check(cli.Get(name, b))
		fmt.Printf("a[0] = %d\n", a[0])
	}
}
