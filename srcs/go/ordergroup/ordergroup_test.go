package ordergroup

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func genExecOrder(n int) []int {
	var ranks []int
	for i := 0; i < n; i++ {
		ranks = append(ranks, i)
	}
	return ranks
}

func genArriveOrder(n int) []int {
	var ranks []int
	for i := 0; i < n; i++ {
		ranks = append(ranks, n-1-i)
	}
	return ranks
}

func Test_1(t *testing.T) {
	n := 10
	arrives := genArriveOrder(n)

	var execOrder []int
	var lock sync.Mutex
	g := New(n, Option{})
	for _, rank := range arrives {
		fmt.Printf("%d arrived\n", rank)
		func(rank int) {
			g.DoRank(rank, func() {
				fmt.Printf("doing %d\n", rank)
				time.Sleep(5 * time.Millisecond)
				lock.Lock()
				defer lock.Unlock()
				execOrder = append(execOrder, rank)
			})
		}(rank)
	}
	g.Wait()
	scheduledOrder := genExecOrder(n)
	if !eq(scheduledOrder, execOrder) {
		t.Errorf("unexpected execOrder: %q, want: %q", execOrder, scheduledOrder)
	}
	g.Stop()
}

func eq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, s := range a {
		if s != b[i] {
			return false
		}
	}
	return true
}
