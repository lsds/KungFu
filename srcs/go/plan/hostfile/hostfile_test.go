package hostfile

import (
	"testing"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func Test_Parse(t *testing.T) {
	text := `
	# ...
	192.168.0.3 slots=4 # ...
	# ...
   	127.0.0.1 slots=8 public_addr=x.y.z# ...
	`
	hl, err := Parse(text)
	assert.OK(err)
	assert.True(len(hl) == 2)
	assert.True(hl[0].IPv4 == plan.MustParseIPv4(`192.168.0.3`))
	assert.True(hl[0].Slots == 4)
	assert.True(hl[0].PublicAddr == `192.168.0.3`)
	assert.True(hl[1].IPv4 == plan.MustParseIPv4(`127.0.0.1`))
	assert.True(hl[1].Slots == 8)
	assert.True(hl[1].PublicAddr == `x.y.z`)
}
