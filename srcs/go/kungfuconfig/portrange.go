package kungfuconfig

import (
	"strconv"

	"github.com/lsds/KungFu/srcs/go/utils"
)

type PortRange struct {
	begin uint16
}

func parsePortRange(val string) PortRange {
	p, err := strconv.Atoi(val)
	if err != nil {
		utils.ExitErr(err)
	}
	return PortRange{
		begin: uint16(p),
	}
}

func (r PortRange) Get(i int) uint16 {
	return r.begin + uint16(i)
}
