package main

import "github.com/lsds/KungFu/srcs/go/cmd/kungfu-run/app"

import "C"

//export GoKungfuRunMain
func GoKungfuRunMain() { app.Main() }
