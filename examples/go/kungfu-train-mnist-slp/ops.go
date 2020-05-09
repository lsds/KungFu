package main

import (
	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func smm(m, n, k int, xs, ys, zs []float32) {
	// xs :: (m , k)
	// ys :: (k , n)
	// zs :: (m , n)
	a := func(i, j int) int { return i*k + j }
	b := func(i, j int) int { return i*n + j }
	c := func(i, j int) int { return i*n + j }
	// TODO: call sgemm
	for i := 0; i < m; i++ {
		for l := 0; l < k; l++ {
			xik := xs[a(i, l)]
			for j := 0; j < n; j++ {
				zs[c(i, j)] += xik * ys[b(l, j)]
			}
		}
	}
}

func Matmul(x, y *Tensor) *Tensor {
	assert.True(x.data.Type == base.F32)
	assert.True(y.data.Type == base.F32)
	xdims := x.shape.Dims()
	ydims := y.shape.Dims()
	assert.True(len(xdims) == 2)
	assert.True(len(ydims) == 2)
	m, k := xdims[0], xdims[1]
	n := ydims[1]
	assert.True(xdims[1] == ydims[0])
	z := NewTensor(Shape{dims: []int{m, n}}, base.F32)
	smm(m, n, k, x.data.AsF32(), y.data.AsF32(), z.data.AsF32())
	return z
}

func addF32(xs, ys, zs []float32) {
	for i, x := range xs {
		zs[i] = x + ys[i]
	}
}

func AddBias(x, y *Tensor) *Tensor {
	assert.True(x.data.Type == base.F32)
	assert.True(y.data.Type == base.F32)
	xdims := x.shape.Dims()
	ydims := y.shape.Dims()
	assert.True(len(xdims) == 2)
	assert.True(len(ydims) == 1)
	n, k := xdims[0], xdims[1]
	assert.True(ydims[0] == k)
	z := NewTensor(Shape{dims: []int{n, k}}, base.F32)
	xs := x.data.AsF32()
	ys := y.data.AsF32()
	zs := z.data.AsF32()
	for i := 0; i < n; i++ {
		addF32(xs[k*i:k*(i+1)], ys, zs[k*i:k*(i+1)])
	}
	return z
}

func argmaxF32(xs []float32) int {
	var idx int
	mx := xs[idx]
	for i, x := range xs {
		if x > mx {
			mx = x
			idx = i
		}
	}
	return idx
}

func Argmax(x *Tensor, dtype base.DataType) *Tensor {
	assert.True(dtype == base.U8)
	assert.True(x.data.Type == base.F32)
	xdims := x.shape.Dims()
	assert.True(len(xdims) == 2)
	n, k := xdims[0], xdims[1]
	z := NewTensor(Shape{dims: []int{n}}, dtype)
	xs := x.data.AsF32()
	zs := z.data.AsU8()
	for i := range zs {
		zs[i] = uint8(argmaxF32(xs[k*i : k*(i+1)]))
	}
	return z
}

func HammingDistance(x, y *Tensor) int {
	assert.True(x.data.Type == base.U8)
	assert.True(y.data.Type == base.U8)
	assert.True(x.shape.Size() == y.shape.Size())
	xs := x.data.AsU8()
	ys := y.data.AsU8()
	var d int
	for i, x := range xs {
		if ys[i] != x {
			d++
		}
	}
	return d
}

func AXPY(a float32, x, y *Tensor, z *Tensor) {
	assert.True(x.data.Type == base.F32)
	assert.True(y.data.Type == base.F32)
	assert.True(z.data.Type == base.F32)
	assert.True(x.shape.Size() == y.shape.Size())
	assert.True(x.shape.Size() == z.shape.Size())
	xs := x.data.AsF32()
	ys := y.data.AsF32()
	zs := z.data.AsF32()
	for i, x := range xs {
		zs[i] = a*x + ys[i]
	}
}
