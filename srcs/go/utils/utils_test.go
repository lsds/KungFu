package utils

import (
	"context"
	"testing"

	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func Test_Poll_OK(t *testing.T) {
	var n int
	f := func() bool {
		n++
		return n > 3
	}
	failed, ok := Poll(context.TODO(), f)
	assert.True(ok)
	assert.True(failed == 3)
}

func Test_Poll_Fail(t *testing.T) {
	ctx, cancel := context.WithCancel(context.TODO())
	var n int
	f := func() bool {
		n++
		if n == 2 {
			cancel()
		}
		return n > 3
	}
	failed, ok := Poll(ctx, f)
	assert.True(!ok)
	assert.True(failed == 2)
}
