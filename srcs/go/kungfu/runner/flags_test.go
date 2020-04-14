package runner

import (
	"flag"
	"testing"

	"github.com/lsds/KungFu/srcs/go/plan"
)

func Test_flag(t *testing.T) {
	var f FlagSet

	testParse := func(args []string) {
		fs := flag.NewFlagSet("", flag.ContinueOnError)
		f.Register(fs)
		if err := fs.Parse(args); err != nil {
			t.Error(err)
		}
	}
	{
		args := []string{`-port-range`, `8080-8088`}
		testParse(args)
		pr := plan.PortRange{Begin: 8080, End: 8088}
		if pr != f.PortRange {
			t.Error("failed to parse -port-range")
		}
	}
}
