package store

import (
	"testing"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

func Test_1(t *testing.T) {
	vs := NewVersionedStore(1)

	dtype := kb.KungFu_UINT8
	a := kb.NewBuffer(1, dtype)
	b := kb.NewBuffer(1, dtype)

	a.Data[0] = 1
	b.Data[0] = 2

	vs.Create("0xff", "a.idx", a)
	if err := vs.Commit("0xff", "a.idx", a); err == nil {
		t.Error("Commit should return error")
	}

	if err := vs.Checkout("0x00", "a.idx", b); err == nil {
		t.Error("Checkout should return error")
	}
	vs.Checkout("0xff", "a.idx", b)

	if b.Data[0] != 1 {
		t.Error("Checkout failed")
	}

	vs.Create("0x100", "a.idx", a)
	if err := vs.Checkout("0xff", "a.idx", b); err == nil {
		t.Error("Checkout should return error")
	}
}
