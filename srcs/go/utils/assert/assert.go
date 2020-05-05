package assert

import (
	"fmt"
	"os"
	"runtime"
)

func perror(name, loc string) {
	fmt.Fprintf(os.Stderr, "%s failed at %s\n", name, loc)
}

func OK(err error) {
	if err != nil {
		pc, fn, line, _ := runtime.Caller(1)
		loc := fmt.Sprintf("%v:%s:%d", pc, fn, line)
		perror(`assertOK`, loc)
		os.Exit(1)
	}
}

func True(ok bool) {
	if !ok {
		pc, fn, line, _ := runtime.Caller(1)
		loc := fmt.Sprintf("%v:%s:%d", pc, fn, line)
		perror(`assertTrue`, loc)
		os.Exit(1)
	}
}
