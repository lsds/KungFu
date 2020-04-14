package utils

import (
	"errors"
	"fmt"
	"os"
	"runtime"
)

func ExitErr(err error) {
	pc, fn, line, _ := runtime.Caller(1)
	loc := fmt.Sprintf("%v:%s:%d", pc, fn, line)
	fmt.Printf("exit on error: %v at %s\n", err, loc)
	os.Exit(1)
}

var errImpossible = errors.New("impossible")

func Immpossible() {
	ExitErr(errImpossible)
}

func MergeErrors(errs []error, hint string) error {
	var msg string
	var failed int
	for _, e := range errs {
		if e != nil {
			failed++
			if len(msg) > 0 {
				msg += ", "
			}
			msg += e.Error()
		}
	}
	if failed == 0 {
		return nil
	}
	return fmt.Errorf("%s failed with %s: %s", hint, Pluralize(failed, "error", "errors"), msg)
}
