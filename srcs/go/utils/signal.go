package utils

import (
	"os"
	"os/signal"
	"syscall"
)

func Trap(cancel func(os.Signal)) {
	c := make(chan os.Signal)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		sig := <-c
		cancel(sig)
	}()
}
