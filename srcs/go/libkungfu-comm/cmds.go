package main

import (
	"os"

	elastic_app "github.com/lsds/KungFu/srcs/go/cmd/kungfu-elastic-run/app"
	"github.com/lsds/KungFu/srcs/go/cmd/kungfu-run/app"
)

import "C"

//export GoKungfuRunMain
func GoKungfuRunMain(shiftArgc int) {
	args := os.Args[shiftArgc:] // remove wrapper program name (`which python`)
	app.Main(args)
}

//export GoKungfuElasticRunMain
func GoKungfuElasticRunMain(shiftArgc int) {
	args := os.Args[shiftArgc:] // remove wrapper program name (`which python`)
	elastic_app.Main(args)
}
