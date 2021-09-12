package app

import (
	"flag"
	"fmt"
	"os"
	"strconv"

	runapp "github.com/lsds/KungFu/srcs/go/cmd/kungfu-run/app"
	"github.com/lsds/KungFu/srcs/go/plan"
)

var (
	np        = flag.Int("np", 1, "number of init workers")
	waitInit  = flag.Bool("d", false, "wait to be initialized")
	portRange = flag.String("port-range", plan.DefaultPortRange.String(), "port range")
)

func Main(inputArgs []string) {
	flag.CommandLine.Parse(inputArgs)
	logArgs(inputArgs, `inputArgs`)

	args := []string{os.Args[0]}
	// args = append(args, "-q")
	args = append(args, "-logdir", "logs")
	args = append(args, "-np", strconv.Itoa(*np))
	args = append(args, "-port-range", *portRange)

	args = append(args, `-builtin-config-port`, `9100`)
	args = append(args, `-config-server`, `http://127.0.0.1:9100/config`)

	args = append(args, flag.CommandLine.Args()...)

	logArgs(args, `args`)
	run(args)
}

func logArgs(args []string, name string) {
	for i, a := range args {
		fmt.Printf("%s[%d]=%s\n", name, i, a)
	}
}

func run(args []string) {
	runapp.Main(args)
}
