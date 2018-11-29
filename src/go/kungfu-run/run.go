package main

import (
	"context"
	"io"
	"os/exec"

	"github.com/luomai/kungfu/src/go/iostream"
	"github.com/luomai/kungfu/src/go/xterm"
)

var (
	warn = xterm.Red
)

func run(ctx context.Context, prefix string, prog string, args []string, envs []string) error {
	cmd := exec.Command(prog, args...)
	cmd.Env = envs
	if stdout, err := cmd.StdoutPipe(); err == nil {
		go streamPipe(prefix+"stdout", stdout)
	} else {
		return err
	}
	if stderr, err := cmd.StderrPipe(); err == nil {
		go streamPipe(prefix+warn.S("stderr"), stderr)
	} else {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	done := make(chan error)
	go func() { done <- cmd.Wait() }()
	select {
	case <-ctx.Done():
		cmd.Process.Kill()
		return ctx.Err()
	case err := <-done:
		return err
	}
}

func streamPipe(name string, r io.Reader) error {
	w := iostream.NewLogWriter(name)
	return iostream.Tee(r, w)
}
