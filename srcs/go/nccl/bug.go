package nccl

import (
	"fmt"
	"math/rand"
	"os"
	"time"
)

const Bug = `Inconsistency detected by ld.so`

func init() {
	rand.Seed(time.Now().Unix())
}

func RandomFailure() {
	n := rand.Int() % 10
	fmt.Printf("n=%d\n", n)
	if n > 1 {
		fmt.Fprintf(os.Stderr, "%s", Bug)
		os.Exit(1)
	}
	fmt.Printf("lucky!\n")
}
