package main

import (
	"bytes"
	"fmt"

	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

type Shape struct {
	dims []int
}

func (s Shape) Size() int {
	d := 1
	for _, dim := range s.dims {
		d *= dim
	}
	return d
}

func (s Shape) Rank() int {
	return len(s.dims)
}

func (s Shape) Dims() []int {
	return s.dims[:]
}

func (s Shape) SubShape() Shape {
	return Shape{dims: s.dims[1:]}
}

func (s Shape) String() string {
	b := &bytes.Buffer{}
	fmt.Fprintf(b, "(")
	for i, d := range s.dims {
		if i > 0 {
			fmt.Fprintf(b, ",")
		}
		fmt.Fprintf(b, "%d", d)
	}
	fmt.Fprintf(b, ")")
	return b.String()
}

type Tensor struct {
	shape Shape
	data  *base.Vector
}

func NewTensor(shape Shape, dtype base.DataType) *Tensor {
	data := base.NewVector(shape.Size(), dtype)
	return &Tensor{
		shape: shape,
		data:  data,
	}
}

// Ldm returns the leading dimension
func (t *Tensor) Ldm() int {
	assert.True(len(t.shape.dims) > 0)
	return t.shape.dims[0]
}

func (t *Tensor) Shape() Shape {
	return t.shape
}

func (t *Tensor) Data() []byte {
	return t.data.Data
}

func (t *Tensor) Info() string {
	return fmt.Sprintf("%s%s", t.data.Type, t.shape)
}

func (t *Tensor) Slice(i, j int) *Tensor {
	subShape := t.shape.SubShape()
	shape := Shape{dims: append([]int{j - i}, subShape.dims...)}
	m := subShape.Size()
	return &Tensor{
		shape: shape,
		data:  t.data.Slice(m*i, m*j),
	}
}

func (t *Tensor) Reshape(shape Shape) *Tensor {
	assert.True(shape.Size() == t.shape.Size())
	return &Tensor{
		shape: shape,
		data:  t.data,
	}
}

func (t *Tensor) Cast(dtype base.DataType) *Tensor {
	assert.True(t.data.Type == base.U8) // TODO: support more types
	assert.True(dtype == base.F32)      // TODO: support more types
	s := NewTensor(t.shape, dtype)
	xs := t.data.AsU8()
	ys := s.data.AsF32()
	for i, x := range xs {
		ys[i] = float32(x)
	}
	return s
}

func (t *Tensor) DivBy(y float32) {
	assert.True(t.data.Type == base.F32) // TODO: support more types
	xs := t.data.AsF32()
	for i, x := range xs {
		xs[i] = x / y
	}
}
