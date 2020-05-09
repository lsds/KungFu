package main

import (
	"github.com/lsds/KungFu/srcs/go/kungfu/base"
)

type BinaryResult struct{ succ, fail int }

func (r *BinaryResult) Add(s BinaryResult) {
	r.succ += s.succ
	r.fail += s.fail
}

func (r *BinaryResult) Accuracy() float32 {
	tot := r.succ + r.fail
	if tot == 0 {
		return 0
	}
	return float32(r.succ) / float32(tot)
}

type GradVar struct {
	G *Tensor
	V *Tensor
}

type Model interface {
	train(samples, labels *Tensor) []*GradVar
	test(samples, labels *Tensor) BinaryResult
}

type SLP struct {
	inputSize int
	weight    *Tensor
	bias      *Tensor
}

/*
	l = H(y', softmax(x * w + b))
*/
func (s *SLP) train(samples, labels *Tensor) []*GradVar {
	samples = samples.Reshape(Shape{dims: []int{samples.Ldm(), s.inputSize}})
	Matmul(samples, s.weight)
	return []*GradVar{
		{
			G: nil,
			V: s.weight,
		},
		{
			G: nil,
			V: s.bias,
		},
	}
}

func (s *SLP) test(samples, labels *Tensor) BinaryResult {
	samples = samples.Reshape(Shape{dims: []int{samples.Ldm(), s.inputSize}})
	y := Matmul(samples, s.weight)
	y = AddBias(y, s.bias)
	y = Argmax(y, base.U8)
	diff := HammingDistance(y, labels)
	return BinaryResult{
		succ: y.Ldm() - diff,
		fail: diff,
	}
}

func createSLP(inputSize int, logits int) Model {
	return &SLP{
		inputSize: inputSize,
		weight:    NewTensor(Shape{dims: []int{inputSize, logits}}, base.F32),
		bias:      NewTensor(Shape{dims: []int{logits}}, base.F32),
	}
}
