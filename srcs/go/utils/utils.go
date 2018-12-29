package utils

import (
	"log"
	"os"
	"strings"
	"time"
)

func LogArgs() {
	for i, a := range os.Args {
		log.Printf("args[%d]=%s", i, a)
	}
}

func LogKungfuEnv() {
	for _, kv := range os.Environ() {
		if strings.HasPrefix(kv, `KUNGFU_`) {
			log.Printf("env: %s", kv)
		}
	}
}

func ExitErr(err error) {
	log.Printf("exit on error: %v", err)
	os.Exit(1)
}

func Measure(f func() error) (time.Duration, error) {
	t0 := time.Now()
	err := f()
	d := time.Since(t0)
	return d, err
}
