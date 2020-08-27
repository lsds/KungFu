package graph

import (
	"testing"

	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func Test_FromForestArrayI32(t *testing.T) {
	{
		f := []int32{1, 1, 1}
		g, m, ok := FromForestArrayI32(f)
		assert.True(g != nil)
		assert.True(ok)
		assert.True(m == 1)
	}
	{
		f := []int32{0, 1, 2}
		g, m, ok := FromForestArrayI32(f)
		assert.True(g != nil)
		assert.True(ok)
		assert.True(m == 3)
	}
	{
		f := []int32{1, 2, 3}
		g, m, ok := FromForestArrayI32(f)
		assert.True(g == nil)
		assert.True(!ok)
		assert.True(m == 0)
	}
}
