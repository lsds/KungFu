package utils

import (
	"fmt"
	"strconv"
	"time"
)

var (
	// -ldflags "-X github.com/lsds/KungFu/srcs/go/utils.buildtimeString=$bt
	buildtimeString string

	buildtime int64
)

func init() {
	buildtime, _ = strconv.ParseInt(buildtimeString, 10, 64)
}

func ShowBuildInfo() {
	if len(buildtimeString) == 0 {
		fmt.Printf("unknown build time\n")
		return
	}
	bt := time.Unix(buildtime, 0)
	fmt.Printf("built %s ago\n", time.Since(bt))
}
