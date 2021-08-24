package main

import (
	"os"

	"github.com/lsds/KungFu/srcs/go/cmd/kungfu-run/app"
)

import "C"

//export GoKungfuRunMain
func GoKungfuRunMain(shiftArgc int) {
	args := os.Args[shiftArgc:] // remove wrapper program name (`which python`)
	app.Main(args)
}
