package hostfile

import (
	"testing"

	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func Test_Parse(t *testing.T) {
	text := `
	# ...
	127.0.0.1 slots=4 # ...
	# ...
   	127.0.0.1 slots=8 # ...
	`
	hl, err := Parse(text)
	assert.OK(err)
	assert.True(len(hl) == 2)
	assert.True(hl[0].Slots == 4)
	assert.True(hl[1].Slots == 8)
}
