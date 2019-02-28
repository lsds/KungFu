package ordergroup

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func genExecOrder(n int) []string {
	var names []string
	for i := 0; i < n; i++ {
		names = append(names, nameFor(i))
	}
	return names
}

func genArriveOrder(n int) []string {
	var names []string
	for i := 0; i < n; i++ {
		names = append(names, nameFor(n-1-i))
	}
	return names
}

func nameFor(i int) string {
	return fmt.Sprintf("%d-th-name", i)
}

func Test_1(t *testing.T) {
	n := 10
	scheduledOrder := genExecOrder(n)
	arrives := genArriveOrder(n)

	var execOrder []string
	var lock sync.Mutex
	g := New(scheduledOrder)
	for _, name := range arrives {
		fmt.Printf("%s arrived\n", name)
		func(name string) {
			g.Do(name, func() {
				fmt.Printf("doing %s\n", name)
				time.Sleep(5 * time.Millisecond)
				lock.Lock()
				defer lock.Unlock()
				execOrder = append(execOrder, name)
			})
		}(name)
	}
	g.Wait()
	if !eq(scheduledOrder, execOrder) {
		t.Errorf("unexpected execOrder: %q, want: %q", execOrder, scheduledOrder)
	}
}

func eq(a, b []string) bool {
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
