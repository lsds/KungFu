package assert

import (
	"fmt"
	"os"
	"runtime"
)

func OK(err error) {
	if err != nil {
		pc, fn, line, _ := runtime.Caller(1)
		loc := fmt.Sprintf("%v:%s:%d", pc, fn, line)
		fmt.Printf("assertOK failed at %s", loc)
		os.Exit(1)
	}
}
